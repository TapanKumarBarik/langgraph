# services/memory_service.py
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path

from azure.storage.blob.aio import BlobServiceClient
from azure.search.documents.aio import SearchClient
from azure.core.credentials import AzureKeyCredential
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
import structlog

logger = structlog.get_logger()

class ConversationMemoryService:
    def __init__(
        self,
        use_local_storage: bool = False,
        connection_string: str = "",
        container_name: str = "conversations",
        local_storage_path: str = "./local_storage/conversations"
    ):
        self.use_local_storage = use_local_storage
        self.connection_string = connection_string
        self.container_name = container_name
        self.local_storage_path = local_storage_path
        self.blob_service_client = None
        self.container_client = None
        self.conversations = {}  # In-memory cache for conversations

    async def initialize(self):
        """Initialize storage service"""
        try:
            if self.use_local_storage:
                os.makedirs(self.local_storage_path, exist_ok=True)
                logger.info("Local storage initialized", path=self.local_storage_path)
            else:
                logger.info("Azure Blob Storage initialized")
        except Exception as e:
            logger.error("Failed to initialize storage", error=str(e))
            raise

    async def save_conversation_turn(self, user_id: str, conversation_id: str, 
                                   human_message: BaseMessage, ai_message: BaseMessage,
                                   metadata: Dict[str, Any] = None) -> None:
        """Save a conversation turn"""
        try:
            conversation_key = f"{user_id}:{conversation_id}"
            
            if conversation_key not in self.conversations:
                self.conversations[conversation_key] = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "messages": [],
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            
            # Add messages
            self.conversations[conversation_key]["messages"].extend([
                {
                    "type": "human",
                    "content": human_message.content,
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "type": "ai", 
                    "content": ai_message.content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": metadata or {}
                }
            ])
            
            self.conversations[conversation_key]["updated_at"] = datetime.utcnow().isoformat()
            
            # Persist to storage
            await self._persist_conversation(user_id, conversation_id, self.conversations[conversation_key])
            
        except Exception as e:
            logger.error("Failed to save conversation turn", error=str(e))
            raise

    async def get_conversation_history(self, user_id: str, conversation_id: str) -> List[BaseMessage]:
        """Get conversation history as BaseMessage objects"""
        try:
            conversation_key = f"{user_id}:{conversation_id}"
            
            # Try cache first
            if conversation_key in self.conversations:
                conversation = self.conversations[conversation_key]
            else:
                # Load from storage
                conversation = await self._load_conversation(user_id, conversation_id)
                if conversation:
                    self.conversations[conversation_key] = conversation
            
            if not conversation:
                return []
            
            # Convert to BaseMessage objects
            messages = []
            for msg in conversation.get("messages", []):
                if msg["type"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            return messages
            
        except Exception as e:
            logger.error("Failed to get conversation history", error=str(e))
            return []

    async def list_user_conversations(self, user_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """List conversations for a specific user"""
        try:
            # Load all conversations for user from storage if needed
            await self._load_user_conversations(user_id)
            
            # Filter conversations for this user
            user_conversations = []
            for key, conv in self.conversations.items():
                if conv.get("user_id") == user_id:
                    user_conversations.append({
                        "conversation_id": conv["conversation_id"],
                        "created_at": conv["created_at"],
                        "updated_at": conv["updated_at"],
                        "message_count": len(conv.get("messages", [])),
                        "last_message": conv.get("messages", [])[-1]["content"][:100] + "..." if conv.get("messages") else ""
                    })
            
            # Sort by updated_at descending
            user_conversations.sort(key=lambda x: x["updated_at"], reverse=True)
            
            # Apply pagination
            return user_conversations[offset:offset + limit]
            
        except Exception as e:
            logger.error("Failed to list user conversations", user_id=user_id, error=str(e))
            return []

    async def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        """Delete a conversation"""
        try:
            conversation_key = f"{user_id}:{conversation_id}"
            
            # Remove from cache
            if conversation_key in self.conversations:
                del self.conversations[conversation_key]
            
            # Remove from storage
            if self.use_local_storage:
                file_path = Path(self.local_storage_path) / user_id / f"{conversation_id}.json"
                if file_path.exists():
                    file_path.unlink()
                    
        except Exception as e:
            logger.error("Failed to delete conversation", error=str(e))
            raise

    async def _persist_conversation(self, user_id: str, conversation_id: str, conversation: Dict[str, Any]) -> None:
        """Persist conversation to storage"""
        if self.use_local_storage:
            user_dir = Path(self.local_storage_path) / user_id
            user_dir.mkdir(exist_ok=True)
            
            file_path = user_dir / f"{conversation_id}.json"
            with open(file_path, 'w') as f:
                json.dump(conversation, f, indent=2)
        else:
            # Implement Azure Blob Storage persistence
            pass

    async def _load_conversation(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation from storage"""
        if self.use_local_storage:
            file_path = Path(self.local_storage_path) / user_id / f"{conversation_id}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        else:
            # Implement Azure Blob Storage loading
            pass
        return None

    async def _load_user_conversations(self, user_id: str) -> None:
        """Load all conversations for a user into cache"""
        if self.use_local_storage:
            user_dir = Path(self.local_storage_path) / user_id
            if user_dir.exists():
                for file_path in user_dir.glob("*.json"):
                    conversation_id = file_path.stem
                    conversation_key = f"{user_id}:{conversation_id}"
                    
                    if conversation_key not in self.conversations:
                        conversation = await self._load_conversation(user_id, conversation_id)
                        if conversation:
                            self.conversations[conversation_key] = conversation

    async def search_documents(
        self, 
        query: str, 
        top: int = 10, 
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search documents in Azure AI Search"""
        try:
            search_params = {
                "search_text": query,
                "top": top,
                "include_total_count": True,
                "highlight_fields": "content",
                "select": "id,title,content,metadata"
            }
            
            if filters:
                search_params["filter"] = filters
            
            results = await self.search_client.search(**search_params)
            
            documents = []
            async for result in results:
                doc = {
                    "id": result.get("id"),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("@search.score", 0),
                    "highlights": result.get("@search.highlights", {}),
                    "metadata": result.get("metadata", {})
                }
                documents.append(doc)
            
            logger.info("Search completed", query=query, results_count=len(documents))
            return documents
            
        except Exception as e:
            logger.error("Search failed", error=str(e), query=query)
            raise
    
    def extract_images_from_results(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract image data from search results"""
        images = []
        return images
    
    def create_citations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create citation information from search results"""
        citations = []
        for i, doc in enumerate(documents):
            citation = {
                "id": f"citation_{i+1}",
                "title": doc.get("title", f"Document {i+1}"),
                "source": doc.get("metadata", {}).get("source", "Azure AI Search"),
                "score": doc.get("score", 0),
                "snippet": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            }
            citations.append(citation)
        return citations

class AzureSQLService:
    def __init__(self, connection_string: str, use_local: bool = False):
        self.connection_string = connection_string
        self.use_local = use_local
        self.engine = None
        self.session_factory = None
        
        # SQL table schemas (configurable)
        self.table_schemas = {
            "customers": ["id", "name", "email", "created_date", "status"],
            "orders": ["id", "customer_id", "product_id", "quantity", "price", "order_date"],
            "products": ["id", "name", "category", "price", "stock_quantity"],
            "sales": ["id", "product_id", "sales_date", "quantity", "revenue"],
            "employees": ["id", "name", "department", "position", "hire_date"],
            "departments": ["id", "name", "manager_id", "budget"],
            "projects": ["id", "name", "department_id", "start_date", "end_date", "status"],
            "inventory": ["id", "product_id", "location", "quantity", "last_updated"],
            "suppliers": ["id", "name", "contact_email", "phone", "address"],
            "transactions": ["id", "account_id", "transaction_date", "amount", "type"],
            "accounts": ["id", "customer_id", "account_type", "balance", "created_date"],
            "reports": ["id", "report_type", "generated_date", "data", "created_by"],
            "metrics": ["id", "metric_name", "value", "date_recorded", "category"],
            "logs": ["id", "event_type", "timestamp", "user_id", "description"],
            "settings": ["id", "setting_name", "setting_value", "updated_by", "updated_date"],
            "user_activity": ["id", "user_id", "activity_type", "timestamp", "details"],
            "feedback": ["id", "customer_id", "rating", "comment", "submitted_date"],
            "campaigns": ["id", "name", "start_date", "end_date", "budget", "status"],
            "analytics": ["id", "metric_type", "dimension", "value", "date"],
            "notifications": ["id", "user_id", "message", "sent_date", "status"]
        }
    
    async def initialize(self):
        """Initialize SQL connection"""
        try:
            if self.use_local:
                # For local SQL Server Express
                self.engine = create_async_engine(
                    self.connection_string,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
            else:
                # For Azure SQL
                async_conn_str = self.connection_string.replace(
                    "mssql+pyodbc://", "mssql+aiodbc://"
                )
                self.engine = create_async_engine(
                    async_conn_str,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
            
            self.session_factory = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("SQL service initialized", local=self.use_local)
            
        except Exception as e:
            logger.error("Failed to initialize SQL", error=str(e))
            raise
    
    async def close(self):
        """Close SQL connections"""
        if self.engine:
            await self.engine.dispose()
    
    def _sanitize_sql_query(self, query: str) -> str:
        """Basic SQL injection protection"""
        dangerous_keywords = [
            "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", 
            "EXEC", "EXECUTE", "sp_", "xp_", "TRUNCATE"
        ]
        
        upper_query = query.upper()
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
        
        return query
    
    async def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Execute SQL query safely"""
        try:
            safe_query = self._sanitize_sql_query(query)
            
            if "TOP" not in safe_query.upper() and "LIMIT" not in safe_query.upper():
                if safe_query.strip().upper().startswith("SELECT"):
                    safe_query = safe_query.replace("SELECT", f"SELECT TOP {max_results}", 1)
            
            async with self.session_factory() as session:
                result = await session.execute(text(safe_query), params or {})
                rows = result.fetchall()
                
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in rows]
                
                logger.info("SQL query executed", 
                           query_preview=safe_query[:100],
                           results_count=len(data))
                
                return data
                
        except Exception as e:
            logger.error("SQL query failed", error=str(e), query=query[:100])
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        return {
            "table_name": table_name,
            "columns": self.table_schemas.get(table_name, []),
            "sample_query": f"SELECT TOP 5 * FROM {table_name}"
        }
    
    def get_all_tables_info(self) -> List[Dict[str, Any]]:
        """Get information about all available tables"""
        return [self.get_table_info(table) for table in self.table_schemas.keys()]

    def get_adventureworks_sample_queries(self) -> List[str]:
        """Get sample queries for AdventureWorks database"""
        return [
            "SELECT TOP 10 * FROM Person.Person",
            "SELECT COUNT(*) FROM Sales.SalesOrderHeader",
            "SELECT ProductID, Name, ListPrice FROM Production.Product WHERE ListPrice > 100"
        ]