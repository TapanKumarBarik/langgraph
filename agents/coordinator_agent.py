# agents/coordinator_agent.py
import json
from typing import Dict, Any, List
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import structlog

logger = structlog.get_logger()

class CoordinatorAgent:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are a coordinator agent that routes user queries to the appropriate specialized agent.

Available agents:
1. "search" - For questions that need information from documents, knowledge base, or web content
2. "sql" - For questions about data, analytics, reports, metrics, or anything requiring database queries
3. "out_of_scope" - For questions outside the scope of available data sources

Analyze the user query and conversation history to determine:
1. Which agent should handle this query
2. Any additional context needed for processing

Consider the conversation flow and user intent. For follow-up questions, maintain context from previous exchanges.

Respond with a JSON object containing:
{
    "agent": "search|sql|out_of_scope",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of routing decision",
    "context": {
        "query_type": "factual|analytical|conversational",
        "requires_history": true|false,
        "data_sources": ["search", "sql", "both"]
    }
}"""
    
    async def route_query(
        self, 
        query: str, 
        conversation_history: List[BaseMessage]
    ) -> Dict[str, Any]:
        """Route query to appropriate agent"""
        try:
            # Prepare context from conversation history
            history_context = ""
            if conversation_history:
                recent_messages = conversation_history[-6:]  # Last 3 exchanges
                history_context = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in recent_messages
                ])
            
            prompt = f"""Current query: "{query}"

Recent conversation history:
{history_context}

Route this query to the appropriate agent."""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                routing_decision = json.loads(response.content)
                logger.info("Query routed", 
                           agent=routing_decision.get("agent"),
                           confidence=routing_decision.get("confidence"))
                return routing_decision
            except json.JSONDecodeError:
                # Fallback parsing
                content = response.content.lower()
                if "search" in content:
                    agent = "search"
                elif "sql" in content:
                    agent = "sql"
                else:
                    agent = "out_of_scope"
                
                return {
                    "agent": agent,
                    "confidence": 0.5,
                    "reasoning": "Fallback routing based on keyword detection",
                    "context": {"query_type": "unknown"}
                }
            
        except Exception as e:
            logger.error("Routing failed", error=str(e))
            return {
                "agent": "out_of_scope",
                "confidence": 0.0,
                "reasoning": f"Error in routing: {str(e)}",
                "context": {}
            }