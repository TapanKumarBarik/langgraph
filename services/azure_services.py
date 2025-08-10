# services/azure_services.py
import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import base64
from datetime import datetime

from azure.search.documents.aio import SearchClient
from azure.core.credentials import AzureKeyCredential
import pyodbc
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import structlog

logger = structlog.get_logger()

class LocalSearchService:
    """Local search service implementation"""
    
    def __init__(self, index_path: str = "./local_storage/search_index"):
        self.index_path = Path(index_path)
        self.documents = []
        self.index_file = self.index_path / "documents.json"
    
    async def initialize(self):
        """Initialize local search index"""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing documents if available
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            else:
                # Create sample documents for testing
                self.documents = self._create_sample_documents()
                await self._save_index()
            
            logger.info("Local search service initialized", doc_count=len(self.documents))
            
        except Exception as e:
            logger.error("Failed to initialize local search", error=str(e))
            raise
    
    def _create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample documents for local testing"""
        return [
            {
                "id": "doc_1",
                "title": "Getting Started Guide",
                "content": "This is a comprehensive getting started guide for new users. It covers all the basic features and functionality of the system.",
                "metadata": {"source": "documentation", "category": "guide"},
                "image_data": None
            },
            {
                "id": "doc_2", 
                "title": "API Documentation",
                "content": "Complete API documentation with examples and best practices. Includes authentication, rate limiting, and error handling.",
                "metadata": {"source": "documentation", "category": "api"},
                "image_data": None
            },
            {
                "id": "doc_3",
                "title": "Troubleshooting FAQ",
                "content": "Frequently asked questions and common troubleshooting steps. Covers installation issues, configuration problems, and performance optimization.",
                "metadata": {"source": "support", "category": "faq"},
                "image_data": None
            }
        ]
    
    async def _save_index(self):
        """Save documents to index file"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
    
    async def search_documents(
        self, 
        query: str, 
        top: int = 10, 
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search documents using simple text matching"""
        try:
            query_lower = query.lower()
            results = []
            
            for doc in self.documents:
                score = 0
                title_lower = doc.get("title", "").lower()
                content_lower = doc.get("content", "").lower()
                
                # Simple scoring based on keyword matches
                if query_lower in title_lower:
                    score += 10
                if query_lower in content_lower:
                    score += 5
                
                # Count individual word matches
                query_words = query_lower.split()
                for word in query_words:
                    if word in title_lower:
                        score += 3
                    if word in content_lower:
                        score += 1
                
                if score > 0:
                    result_doc = doc.copy()
                    result_doc["@search.score"] = score
                    result_doc["@search.highlights"] = {
                        "content": [content_lower[:200] + "..."] if len(content_lower) > 200 else [content_lower]
                    }
                    results.append(result_doc)
            
            # Sort by score and limit results
            results.sort(key=lambda x: x["@search.score"], reverse=True)
            return results[:top]
            
        except Exception as e:
            logger.error("Local search failed", error=str(e))
            return []

class AzureAISearchService:
    def __init__(self, endpoint: str, api_key: str, index_name: str, use_local: bool = False, local_index_path: str = ""):
        self.endpoint = endpoint
        self.api_key = api_key
        self.index_name = index_name
        self.use_local = use_local
        self.search_client = None
        self.local_search = None
        
        if use_local:
            self.local_search = LocalSearchService(local_index_path)
    
    async def initialize(self):
        """Initialize search service (Azure or Local)"""
        try:
            if self.use_local:
                await self.local_search.initialize()
                logger.info("Local search service initialized")
            else:
                credential = AzureKeyCredential(self.api_key)
                self.search_client = SearchClient(
                    endpoint=self.endpoint,
                    index_name=self.index_name,
                    credential=credential
                )
                logger.info("Azure AI Search service initialized")
        except Exception as e:
            logger.error("Failed to initialize search service", error=str(e))
            raise
    
    async def search_documents(
        self, 
        query: str, 
        top: int = 10, 
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search documents in Azure AI Search or Local Search"""
        try:
            if self.use_local:
                return await self.local_search.search_documents(query, top, filters)
            
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
                       results_count=len(documents),
                       service="azure")
            
            return documents
            
        except Exception as e:
            logger.error("Search failed", error=str(e), query=query)
            raise
    
    def extract_images_from_results(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract image data from search results, handle missing images gracefully"""
        images = []
        
        if not documents:
            return images
            
        try:
            for doc in documents:
                image_data = doc.get("image_data")
                
                if image_data:
                    # Handle base64 encoded images
                    if isinstance(image_data, str) and image_data.strip():
                        images.append(image_data)
                    elif isinstance(image_data, list):
                        # Filter out empty or None values
                        valid_images = [img for img in image_data if img and isinstance(img, str) and img.strip()]
                        images.extend(valid_images)
                        
            logger.debug("Images extracted from search results", 
                        total_docs=len(documents), 
                        images_found=len(images))
                        
        except Exception as e:
            logger.warning("Error extracting images from search results", error=str(e))
            # Return empty list if image extraction fails
            
        return images
    
    def create_citations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create citation information from search results"""
        citations = []
        
        if not documents:
            return citations
            
        try:
            for i, doc in enumerate(documents):
                # Safely get values with defaults
                title = doc.get("title", f"Document {i+1}")
                source = "Local Search" if self.use_local else "Azure AI Search"
                
                # Try to get source from metadata
                if isinstance(doc.get("metadata"), dict):
                    source = doc["metadata"].get("source", source)
                
                content = doc.get("content", "")
                snippet = content[:200] + "..." if len(content) > 200 else content
                
                citation = {
                    "id": f"citation_{i+1}",
                    "title": title,
                    "source": source,
                    "score": doc.get("score", doc.get("@search.score", 0)),
                    "snippet": snippet
                }
                citations.append(citation)
                
        except Exception as e:
            logger.warning("Error creating citations", error=str(e))
            
        return citations

class AzureSQLService:
    def __init__(self, connection_string: str, use_local: bool = False):
        self.connection_string = connection_string
        self.use_local = use_local
        self.engine = None
        self.session_factory = None
        
        # AdventureWorks2022 schema (when using local SQL Server Express)
        if use_local:
            self.table_schemas = {
                # Person schema
                "Person.Person": ["BusinessEntityID", "PersonType", "NameStyle", "Title", "FirstName", "MiddleName", "LastName", "Suffix", "EmailPromotion"],
                "Person.EmailAddress": ["BusinessEntityID", "EmailAddressID", "EmailAddress"],
                "Person.Address": ["AddressID", "AddressLine1", "AddressLine2", "City", "StateProvinceID", "PostalCode", "SpatialLocation"],
                "Person.StateProvince": ["StateProvinceID", "StateProvinceCode", "CountryRegionCode", "Name", "TerritoryID"],
                
                # Sales schema
                "Sales.Customer": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "ModifiedDate"],
                "Sales.SalesOrderHeader": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "CustomerID", "SalesPersonID", "TerritoryID", "TotalDue"],
                "Sales.SalesOrderDetail": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal"],
                "Sales.SalesPerson": ["BusinessEntityID", "TerritoryID", "SalesQuota", "Bonus", "CommissionPct", "SalesYTD", "SalesLastYear"],
                "Sales.SalesTerritory": ["TerritoryID", "Name", "CountryRegionCode", "Group", "SalesYTD", "SalesLastYear", "CostYTD", "CostLastYear"],
                
                # Production schema  
                "Production.Product": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate"],
                "Production.ProductCategory": ["ProductCategoryID", "Name"],
                "Production.ProductSubcategory": ["ProductSubcategoryID", "ProductCategoryID", "Name"],
                "Production.ProductInventory": ["ProductID", "LocationID", "Shelf", "Bin", "Quantity"],
                "Production.WorkOrder": ["WorkOrderID", "ProductID", "OrderQty", "StockedQty", "ScrappedQty", "StartDate", "EndDate", "DueDate"],
                
                # HumanResources schema
                "HumanResources.Employee": ["BusinessEntityID", "NationalIDNumber", "LoginID", "OrganizationLevel", "JobTitle", "BirthDate", "MaritalStatus", "Gender", "HireDate", "SalariedFlag", "VacationHours", "SickLeaveHours"],
                "HumanResources.Department": ["DepartmentID", "Name", "GroupName"],
                "HumanResources.EmployeeDepartmentHistory": ["BusinessEntityID", "DepartmentID", "ShiftID", "StartDate", "EndDate"],
                
                # Purchasing schema
                "Purchasing.Vendor": ["BusinessEntityID", "AccountNumber", "Name", "CreditRating", "PreferredVendorStatus", "ActiveFlag"],
                "Purchasing.PurchaseOrderHeader": ["PurchaseOrderID", "RevisionNumber", "Status", "EmployeeID", "VendorID", "ShipMethodID", "OrderDate", "ShipDate", "SubTotal", "TaxAmt", "Freight", "TotalDue"],
                "Purchasing.PurchaseOrderDetail": ["PurchaseOrderID", "PurchaseOrderDetailID", "DueDate", "OrderQty", "ProductID", "UnitPrice", "LineTotal", "ReceivedQty", "RejectedQty", "StockedQty"]
            }
        else:
            # Generic schema for Azure SQL
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
                # For local SQL Server Express, ensure async driver
                if "aiodbc" not in self.connection_string and "asyncio" not in self.connection_string:
                    # Convert pyodbc to aiodbc for async support
                    self.connection_string = self.connection_string.replace("mssql+pyodbc://", "mssql+aiodbc://")
            
            self.engine = create_async_engine(
                self.connection_string,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20
            )
            
            self.session_factory = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            db_type = "Local SQL Server Express (AdventureWorks2022)" if self.use_local else "Azure SQL"
            logger.info("SQL service initialized", database_type=db_type)
            
        except Exception as e:
            logger.error("Failed to initialize SQL service", error=str(e))
            raise
    
    async def close(self):
        """Close SQL connections"""
        if self.engine:
            await self.engine.dispose()
    
    def _sanitize_sql_query(self, query: str) -> str:
        """Enhanced SQL injection protection"""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Remove dangerous keywords
        dangerous_keywords = [
            "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", 
            "EXEC", "EXECUTE", "sp_", "xp_", "TRUNCATE", "MERGE",
            "BULK", "OPENROWSET", "OPENDATASOURCE", "OPENQUERY"
        ]
        
        upper_query = query.upper()
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r';\s*--',  # Comment injection
            r';\s*SELECT',  # Statement chaining
            r'UNION\s+SELECT',  # Union injection
            r'@@\w+',  # System variables
            r'WAITFOR\s+DELAY',  # Time delays
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, upper_query):
                raise ValueError(f"Suspicious SQL pattern detected")
        
        return query.strip()
    
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
            
            # Add TOP clause if not present (for SQL Server)
            if "TOP" not in safe_query.upper() and "LIMIT" not in safe_query.upper():
                if safe_query.strip().upper().startswith("SELECT"):
                    # Insert TOP clause after SELECT
                    safe_query = re.sub(
                        r'^SELECT\s+',
                        f'SELECT TOP {max_results} ',
                        safe_query,
                        flags=re.IGNORECASE
                    )
            
            async with self.session_factory() as session:
                result = await session.execute(text(safe_query), params or {})
                rows = result.fetchall()
                
                # Convert to list of dicts
                if rows:
                    columns = result.keys()
                    data = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i]
                            # Handle datetime objects
                            if hasattr(value, 'isoformat'):
                                value = value.isoformat()
                            row_dict[col] = value
                        data.append(row_dict)
                else:
                    data = []
                
                logger.info("SQL query executed", 
                           query_preview=safe_query[:100] + "..." if len(safe_query) > 100 else safe_query,
                           results_count=len(data),
                           database_type="local" if self.use_local else "azure")
                
                return data
                
        except Exception as e:
            logger.error("SQL query failed", error=str(e), query=query[:100])
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        columns = self.table_schemas.get(table_name, [])
        sample_query = f"SELECT TOP 5 * FROM [{table_name}]" if self.use_local else f"SELECT TOP 5 * FROM {table_name}"
        
        return {
            "table_name": table_name,
            "columns": columns,
            "sample_query": sample_query,
            "database_type": "AdventureWorks2022" if self.use_local else "Generic"
        }
    
    def get_all_tables_info(self) -> List[Dict[str, Any]]:
        """Get information about all available tables"""
        return [self.get_table_info(table) for table in self.table_schemas.keys()]
    
    def get_adventureworks_sample_queries(self) -> List[str]:
        """Get sample queries for AdventureWorks2022"""
        if not self.use_local:
            return []
            
        return [
            "SELECT TOP 10 FirstName, LastName FROM Person.Person",
            "SELECT TOP 10 Name, ListPrice FROM Production.Product WHERE ListPrice > 0 ORDER BY ListPrice DESC",
            "SELECT COUNT(*) as TotalOrders FROM Sales.SalesOrderHeader",
            "SELECT TOP 5 Name, SalesYTD FROM Sales.SalesTerritory ORDER BY SalesYTD DESC",
            "SELECT TOP 10 JobTitle, COUNT(*) as EmployeeCount FROM HumanResources.Employee GROUP BY JobTitle"
        ]