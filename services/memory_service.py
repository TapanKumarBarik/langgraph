# services/memory_service.py
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

from azure.storage.blob.aio import BlobServiceClient
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
import structlog

logger = structlog.get_logger()

class ConversationMemoryService:
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = None
        self.container_client = None
    
    async def initialize(self):
        """Initialize Azure AI Search client"""
        try:
            credential = AzureKeyCredential(self.api_key)
            self.search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=credential
            )
            logger.info("Azure AI Search service initialized")
        except Exception as e:
            logger.error("Failed to initialize Azure AI Search", error=str(e))
            raise
    
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
                "select": "id,title,content,metadata,image_data"
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
                    "metadata": result.get("metadata", {}),
                    "image_data": result.get("image_data")
                }
                documents.append(doc)
            
            logger.info("Search completed", 
                       query=query, 
                       results_count=len(documents))
            
            return documents
            
        except Exception as e:
            logger.error("Search failed", error=str(e), query=query)
            raise
    
    def extract_images_from_results(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract image data from search results"""
        images = []
        for doc in documents:
            if doc.get("image_data"):
                # Handle base64 encoded images
                if isinstance(doc["image_data"], str):
                    images.append(doc["image_data"])
                elif isinstance(doc["image_data"], list):
                    images.extend(doc["image_data"])
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
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
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
            # Convert connection string for async usage
            async_conn_str = self.connection_string.replace(
                "Driver={ODBC Driver", "Driver={ODBC Driver"
            )
            if "asyncio" not in async_conn_str:
                async_conn_str = async_conn_str.replace(
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
            
            logger.info("Azure SQL service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize Azure SQL", error=str(e))
            raise
    
    async def close(self):
        """Close SQL connections"""
        if self.engine:
            await self.engine.dispose()
    
    def _sanitize_sql_query(self, query: str) -> str:
        """Basic SQL injection protection"""
        # Remove dangerous keywords
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
            # Sanitize query
            safe_query = self._sanitize_sql_query(query)
            
            # Add LIMIT if not present
            if "TOP" not in safe_query.upper() and "LIMIT" not in safe_query.upper():
                if safe_query.strip().upper().startswith("SELECT"):
                    safe_query = safe_query.replace("SELECT", f"SELECT TOP {max_results}", 1)
            
            async with self.session_factory() as session:
                result = await session.execute(text(safe_query), params or {})
                rows = result.fetchall()
                
                # Convert to list of dicts
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
