# main.py
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from services.memory_service import ConversationMemoryService
from services.azure_services import AzureAISearchService, AzureSQLService
from agents.coordinator_agent import CoordinatorAgent
from agents.search_agent import SearchAgent
from agents.sql_agent import SQLAgent
from agents.followup_agent import FollowupAgent
from utils.retry_decorator import retry_with_exponential_backoff
from utils.validation import validate_input
from config.settings import Settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Pydantic Models
class ChatMessage(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    user_id: str = Field(..., min_length=1, max_length=100)
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    
    @validator('message')
    def validate_message(cls, v):
        return validate_input(v)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    agent_used: str
    sources: List[Dict[str, Any]] = []
    images: List[str] = []
    followup_suggestions: List[str] = []
    metadata: Dict[str, Any] = {}

class AgentState(BaseModel):
    messages: List[BaseMessage] = []
    user_id: str
    conversation_id: str
    context: Dict[str, Any] = {}
    current_agent: Optional[str] = None
    search_results: List[Dict[str, Any]] = []
    sql_results: List[Dict[str, Any]] = []
    images: List[str] = []
    sources: List[Dict[str, Any]] = []
    followup_suggestions: List[str] = []
    retry_count: int = 0
    error_message: Optional[str] = None

# Global services
settings = Settings()
memory_service: Optional[ConversationMemoryService] = None
azure_search: Optional[AzureAISearchService] = None
azure_sql: Optional[AzureSQLService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global memory_service, azure_search, azure_sql
    
    logger.info("Initializing services...", 
                environment=settings.ENVIRONMENT,
                use_local_storage=settings.USE_LOCAL_STORAGE,
                use_local_search=settings.USE_LOCAL_SEARCH,
                use_local_sql=settings.USE_LOCAL_SQL)
    
    try:
        # Initialize Memory Service
        memory_service = ConversationMemoryService(
            use_local_storage=settings.USE_LOCAL_STORAGE,
            connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
            container_name=settings.CONVERSATION_CONTAINER,
            local_storage_path=settings.LOCAL_STORAGE_PATH
        )
        
        # Initialize Search Service
        azure_search = AzureAISearchService(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            api_key=settings.AZURE_SEARCH_API_KEY,
            index_name=settings.AZURE_SEARCH_INDEX,
            use_local=settings.USE_LOCAL_SEARCH,
            local_index_path=settings.LOCAL_SEARCH_INDEX_PATH
        )
        
        # Initialize SQL Service
        azure_sql = AzureSQLService(
            connection_string=settings.sql_connection_string,
            use_local=settings.USE_LOCAL_SQL
        )
        
        # Initialize all services
        await memory_service.initialize()
        await azure_search.initialize()
        await azure_sql.initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    if azure_sql:
        await azure_sql.close()

app = FastAPI(
    title="Multi-Agent Conversational AI",
    description="Production-grade LangGraph multi-agent system with Azure/Local integrations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MultiAgentOrchestrator:
    def __init__(self):
        # Initialize LLM based on configuration
        if settings.OPENAI_API_TYPE == "openai":
            # For local LLM servers (like Ollama, LM Studio, etc.)
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                base_url=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                temperature=0.1,
                max_tokens=2000
            )
        else:
            # For Azure OpenAI
            self.llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                temperature=0.1,
                max_tokens=2000
            )
        
        # Initialize agents
        self.coordinator_agent = CoordinatorAgent(self.llm)
        self.search_agent = SearchAgent(self.llm, azure_search)
        self.sql_agent = SQLAgent(self.llm, azure_sql)
        
        if settings.ENABLE_FOLLOWUP_SUGGESTIONS:
            self.followup_agent = FollowupAgent(self.llm)
        else:
            self.followup_agent = None
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("sql", self._sql_node)
        
        if self.followup_agent:
            workflow.add_node("followup", self._followup_node)
        
        workflow.add_node("response", self._response_node)
        
        # Add edges
        workflow.set_entry_point("coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self._route_decision,
            {
                "search": "search",
                "sql": "sql",
                "out_of_scope": "response",
                "error": "response"
            }
        )
        
        if self.followup_agent:
            workflow.add_edge("search", "followup")
            workflow.add_edge("sql", "followup")
            workflow.add_edge("followup", "response")
        else:
            workflow.add_edge("search", "response")
            workflow.add_edge("sql", "response")
            
        workflow.add_edge("response", END)
        
        return workflow.compile()
    
    @retry_with_exponential_backoff(max_retries=3)
    async def _coordinator_node(self, state: AgentState) -> AgentState:
        try:
            logger.info("Coordinator processing", conversation_id=state.conversation_id)
            
            # Get conversation history
            history = await memory_service.get_conversation_history(
                state.user_id, state.conversation_id
            )
            
            # Add history to state
            state.messages = history + state.messages
            
            # Determine routing
            routing_decision = await self.coordinator_agent.route_query(
                state.messages[-1].content, history
            )
            
            state.current_agent = routing_decision["agent"]
            state.context.update(routing_decision.get("context", {}))
            
            logger.info("Routing decision made", 
                       agent=state.current_agent, 
                       conversation_id=state.conversation_id)
            
            return state
            
        except Exception as e:
            logger.error("Coordinator node error", error=str(e))
            state.error_message = f"Coordinator error: {str(e)}"
            state.current_agent = "error"
            return state
    
    @retry_with_exponential_backoff(max_retries=3)
    async def _search_node(self, state: AgentState) -> AgentState:
        try:
            logger.info("Search agent processing", conversation_id=state.conversation_id)
            
            query = state.messages[-1].content
            results = await self.search_agent.search_and_process(
                query, state.context
            )
            
            state.search_results = results["documents"]
            state.sources.extend(results["sources"])
            
            # Handle images safely
            if settings.ENABLE_IMAGE_SUPPORT:
                state.images.extend(results.get("images", []))
            
            logger.info("Search completed", 
                       results_count=len(state.search_results),
                       images_count=len(state.images),
                       conversation_id=state.conversation_id)
            
            return state
            
        except Exception as e:
            logger.error("Search node error", error=str(e))
            state.error_message = f"Search error: {str(e)}"
            return state
    
    @retry_with_exponential_backoff(max_retries=3)
    async def _sql_node(self, state: AgentState) -> AgentState:
        try:
            logger.info("SQL agent processing", conversation_id=state.conversation_id)
            
            query = state.messages[-1].content
            results = await self.sql_agent.query_and_process(
                query, state.context
            )
            
            state.sql_results = results["data"]
            state.sources.extend(results["sources"])
            
            logger.info("SQL query completed", 
                       results_count=len(state.sql_results),
                       conversation_id=state.conversation_id)
            
            return state
            
        except Exception as e:
            logger.error("SQL node error", error=str(e))
            state.error_message = f"SQL error: {str(e)}"
            return state
    
    @retry_with_exponential_backoff(max_retries=2)
    async def _followup_node(self, state: AgentState) -> AgentState:
        try:
            if not self.followup_agent:
                return state
                
            logger.info("Followup agent processing", conversation_id=state.conversation_id)
            
            # Generate followup suggestions based on results
            suggestions = await self.followup_agent.generate_followups(
                query=state.messages[-1].content,
                search_results=state.search_results,
                sql_results=state.sql_results,
                context=state.context
            )
            
            state.followup_suggestions = suggestions
            
            return state
            
        except Exception as e:
            logger.error("Followup node error", error=str(e))
            # Followup errors are non-critical
            state.followup_suggestions = []
            return state
    
    async def _response_node(self, state: AgentState) -> AgentState:
        try:
            logger.info("Response node processing", conversation_id=state.conversation_id)
            
            if state.error_message:
                response_content = "I apologize, but I encountered an error while processing your request. Please try again."
            elif state.current_agent == "out_of_scope":
                response_content = "I'm sorry, but your question is outside my scope of knowledge. I can only help with information from our search index and database."
            else:
                # Generate response based on agent type and results
                if state.current_agent == "search":
                    response_content = await self._generate_search_response(state)
                elif state.current_agent == "sql":
                    response_content = await self._generate_sql_response(state)
                else:
                    response_content = "I couldn't determine how to best help you with that question."
            
            # Create AI message
            ai_message = AIMessage(content=response_content)
            state.messages.append(ai_message)
            
            # Save conversation to memory
            await memory_service.save_conversation_turn(
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                human_message=state.messages[-2],
                ai_message=ai_message,
                metadata={
                    "agent": state.current_agent,
                    "sources": state.sources,
                    "images": state.images,
                    "environment": settings.ENVIRONMENT
                }
            )
            
            return state
            
        except Exception as e:
            logger.error("Response node error", error=str(e))
            state.error_message = f"Response generation error: {str(e)}"
            return state
    
    def _route_decision(self, state: AgentState) -> str:
        if state.error_message:
            return "error"
        return state.current_agent or "out_of_scope"
    
    async def _generate_search_response(self, state: AgentState) -> str:
        """Generate response from search results with citations"""
        if not state.search_results:
            return "I couldn't find any relevant information for your query."
        
        context_docs = "\n\n".join([
            f"Source {i+1}: {doc.get('content', '')[:500]}..."
            for i, doc in enumerate(state.search_results[:3])
        ])
        
        # Mention images if present
        image_note = ""
        if state.images and settings.ENABLE_IMAGE_SUPPORT:
            image_note = f"\n\nNote: {len(state.images)} related image(s) have been found and included in the response."
        
        prompt = f"""Based on the following search results, provide a comprehensive answer to the user's question: "{state.messages[-2].content}"

Search Results:
{context_docs}

Guidelines:
- Provide a direct, helpful answer
- Include specific details from the sources
- Mention source numbers in your response for citations
- If images are mentioned in the context, note them in your response{image_note}

Answer:"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    
    async def _generate_sql_response(self, state: AgentState) -> str:
        """Generate response from SQL results"""
        if not state.sql_results:
            return "I couldn't find any relevant data for your query."
        
        # Format SQL results for context
        data_summary = json.dumps(state.sql_results[:10], indent=2, default=str)
        
        db_context = "AdventureWorks2022 database" if settings.USE_LOCAL_SQL else "database"
        
        prompt = f"""Based on the following {db_context} query results, provide a comprehensive answer to the user's question: "{state.messages[-2].content}"

Query Results:
{data_summary}

Guidelines:
- Analyze the data and provide insights
- Present information in a clear, understandable format
- Include specific numbers and details from the results
- If there are trends or patterns, highlight them
- Format data in tables or lists when appropriate

Answer:"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

# Initialize orchestrator
orchestrator = MultiAgentOrchestrator()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """Main chat endpoint for conversational AI"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        logger.info("Chat request received", 
                   user_id=request.user_id,
                   conversation_id=conversation_id,
                   message_length=len(request.message),
                   environment=settings.ENVIRONMENT)
        
        # Create initial state
        state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_id=request.user_id,
            conversation_id=conversation_id,
            context=request.context or {}
        )
        
        # Process through the graph
        final_state = await orchestrator.graph.ainvoke(state)
        
        # Extract response
        response_message = final_state.messages[-1].content
        
        # Log successful completion
        logger.info("Chat request completed", 
                   conversation_id=conversation_id,
                   agent_used=final_state.current_agent,
                   response_length=len(response_message),
                   has_images=len(final_state.images) > 0,
                   has_sources=len(final_state.sources) > 0)
        
        return ChatResponse(
            response=response_message,
            conversation_id=conversation_id,
            agent_used=final_state.current_agent or "unknown",
            sources=final_state.sources,
            images=final_state.images if settings.ENABLE_IMAGE_SUPPORT else [],
            followup_suggestions=final_state.followup_suggestions if settings.ENABLE_FOLLOWUP_SUGGESTIONS else [],
            metadata={
                "processing_time": datetime.utcnow().isoformat(),
                "search_results_count": len(final_state.search_results),
                "sql_results_count": len(final_state.sql_results),
                "environment": settings.ENVIRONMENT,
                "database_type": "AdventureWorks2022" if settings.USE_LOCAL_SQL else "Azure SQL"
            }
        )
        
    except Exception as e:
        logger.error("Chat endpoint error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/conversations/{user_id}/{conversation_id}")
async def get_conversation(user_id: str, conversation_id: str):
    """Get conversation history"""
    try:
        history = await memory_service.get_conversation_history(user_id, conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages": [{"role": "human" if isinstance(msg, HumanMessage) else "ai", 
                         "content": msg.content} for msg in history],
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        logger.error("Get conversation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{user_id}")
async def list_conversations(user_id: str, limit: int = 20, offset: int = 0):
    """List user conversations"""
    try:
        conversations = await memory_service.list_user_conversations(user_id, limit, offset)
        return {
            "conversations": conversations,
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        logger.error("List conversations error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{user_id}/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    """Delete a conversation"""
    try:
        await memory_service.delete_conversation(user_id, conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error("Delete conversation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_configuration():
    """Get current configuration"""
    return {
        "environment": settings.ENVIRONMENT,
        "services": {
            "memory": {
                "type": "local_filesystem" if settings.USE_LOCAL_STORAGE else "azure_blob",
                "path": settings.LOCAL_STORAGE_PATH if settings.USE_LOCAL_STORAGE else "azure_blob"
            },
            "search": {
                "type": "local_search" if settings.USE_LOCAL_SEARCH else "azure_ai_search",
                "index": settings.AZURE_SEARCH_INDEX
            },
            "sql": {
                "type": "local_sql_express" if settings.USE_LOCAL_SQL else "azure_sql",
                "database": settings.LOCAL_SQL_DATABASE if settings.USE_LOCAL_SQL else "azure_sql"
            },
            "llm": {
                "type": settings.OPENAI_API_TYPE,
                "endpoint": settings.AZURE_OPENAI_ENDPOINT,
                "model": settings.AZURE_OPENAI_DEPLOYMENT_NAME
            }
        },
        "features": {
            "image_support": settings.ENABLE_IMAGE_SUPPORT,
            "followup_suggestions": settings.ENABLE_FOLLOWUP_SUGGESTIONS
        }
    }

@app.get("/sql/tables")
async def get_sql_tables():
    """Get available SQL tables and their schemas"""
    try:
        tables_info = azure_sql.get_all_tables_info()
        
        response = {
            "tables": tables_info,
            "database_type": "AdventureWorks2022" if settings.USE_LOCAL_SQL else "Generic Azure SQL",
            "total_tables": len(tables_info)
        }
        
        if settings.USE_LOCAL_SQL:
            response["sample_queries"] = azure_sql.get_adventureworks_sample_queries()
        
        return response
        
    except Exception as e:
        logger.error("Get SQL tables error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT,
        "services": {
            "memory": memory_service is not None,
            "search": azure_search is not None,
            "sql": azure_sql is not None
        },
        "configuration": {
            "local_storage": settings.USE_LOCAL_STORAGE,
            "local_search": settings.USE_LOCAL_SEARCH,
            "local_sql": settings.USE_LOCAL_SQL,
            "image_support": settings.ENABLE_IMAGE_SUPPORT,
            "followup_suggestions": settings.ENABLE_FOLLOWUP_SUGGESTIONS
        }
    }
    
    # Test database connection
    try:
        if azure_sql:
            await azure_sql.execute_query("SELECT 1", max_results=1)
            health_status["database_connection"] = "healthy"
    except Exception as e:
        health_status["database_connection"] = "unhealthy"
        health_status["database_error"] = str(e)
    
    return health_status

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1 if settings.is_local_environment else 4,
        log_level="info",
        access_log=True
    )