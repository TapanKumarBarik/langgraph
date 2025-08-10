# config/settings.py

import os
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class Settings:
    # Environment Configuration
    ENVIRONMENT: str = "azure"
    
    # Azure OpenAI (can be local or Azure)
    AZURE_OPENAI_ENDPOINT: str = "http://localhost:1234/v1"
    AZURE_OPENAI_API_KEY: str = "local-key"
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4"
    OPENAI_API_TYPE: str = "azure"
    
    # Storage Configuration (Azure Blob or Local File System)
    USE_LOCAL_STORAGE: bool = False
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    LOCAL_STORAGE_PATH: str = "./local_storage/conversations"
    CONVERSATION_CONTAINER: str = "conversations"
    
    # Search Configuration (Azure AI Search or Local)
    USE_LOCAL_SEARCH: bool = False
    AZURE_SEARCH_ENDPOINT: str = ""
    AZURE_SEARCH_API_KEY: str = ""
    AZURE_SEARCH_INDEX: str = "knowledge-base"
    LOCAL_SEARCH_INDEX_PATH: str = "./local_storage/search_index"
    
    # SQL Configuration (Azure SQL or Local SQL Server Express)
    USE_LOCAL_SQL: bool = False
    AZURE_SQL_CONNECTION_STRING: str = ""
    LOCAL_SQL_SERVER: str = "localhost\\SQLEXPRESS"
    LOCAL_SQL_DATABASE: str = "AdventureWorks2022"
    LOCAL_SQL_USERNAME: str = ""
    LOCAL_SQL_PASSWORD: str = ""
    LOCAL_SQL_TRUSTED_CONNECTION: bool = True
    LOCAL_SQL_CONNECTION_STRING: str = ""
    
    # Redis (optional for caching)
    REDIS_URL: Optional[str] = None
    USE_REDIS: bool = False
    
    # API Configuration
    MAX_CONVERSATION_HISTORY: int = 20
    MAX_SEARCH_RESULTS: int = 10
    MAX_SQL_RESULTS: int = 50
    
    # Feature Flags
    ENABLE_IMAGE_SUPPORT: bool = True
    ENABLE_FOLLOWUP_SUGGESTIONS: bool = True
    
    def __post_init__(self):
        """Load environment variables and build connection strings"""
        # Load from environment
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "azure")
        
        # Azure OpenAI
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "http://localhost:1234/v1")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "local-key")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "azure")
        
        # Storage
        self.USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "false").lower() == "true"
        self.AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.LOCAL_STORAGE_PATH = os.getenv("LOCAL_STORAGE_PATH", "./local_storage/conversations")
        self.CONVERSATION_CONTAINER = os.getenv("CONVERSATION_CONTAINER", "conversations")
        
        # Search
        self.USE_LOCAL_SEARCH = os.getenv("USE_LOCAL_SEARCH", "false").lower() == "true"
        self.AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
        self.AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
        self.AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "knowledge-base")
        self.LOCAL_SEARCH_INDEX_PATH = os.getenv("LOCAL_SEARCH_INDEX_PATH", "./local_storage/search_index")
        
        # SQL
        self.USE_LOCAL_SQL = os.getenv("USE_LOCAL_SQL", "false").lower() == "true"
        self.AZURE_SQL_CONNECTION_STRING = os.getenv("AZURE_SQL_CONNECTION_STRING", "")
        self.LOCAL_SQL_SERVER = os.getenv("LOCAL_SQL_SERVER", "localhost\\SQLEXPRESS")
        self.LOCAL_SQL_DATABASE = os.getenv("LOCAL_SQL_DATABASE", "AdventureWorks2022")
        self.LOCAL_SQL_USERNAME = os.getenv("LOCAL_SQL_USERNAME", "")
        self.LOCAL_SQL_PASSWORD = os.getenv("LOCAL_SQL_PASSWORD", "")
        self.LOCAL_SQL_TRUSTED_CONNECTION = os.getenv("LOCAL_SQL_TRUSTED_CONNECTION", "true").lower() == "true"
        
        # Redis
        self.REDIS_URL = os.getenv("REDIS_URL")
        self.USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
        
        # API limits
        self.MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
        self.MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
        self.MAX_SQL_RESULTS = int(os.getenv("MAX_SQL_RESULTS", "50"))
        
        # Features
        self.ENABLE_IMAGE_SUPPORT = os.getenv("ENABLE_IMAGE_SUPPORT", "true").lower() == "true"
        self.ENABLE_FOLLOWUP_SUGGESTIONS = os.getenv("ENABLE_FOLLOWUP_SUGGESTIONS", "true").lower() == "true"
        
        # Build local SQL connection string
        if self.USE_LOCAL_SQL:
            self.LOCAL_SQL_CONNECTION_STRING = f"mssql+pyodbc://{self.LOCAL_SQL_SERVER}/{self.LOCAL_SQL_DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&TrustedConnection=yes"
    
    @property
    def sql_connection_string(self) -> str:
        """Get appropriate SQL connection string based on configuration"""
        if self.USE_LOCAL_SQL:
            return self.LOCAL_SQL_CONNECTION_STRING
        return self.AZURE_SQL_CONNECTION_STRING
    
    @property
    def is_local_environment(self) -> bool:
        """Check if running in local environment"""
        return self.ENVIRONMENT == "local"
    
    @property
    def is_hybrid_environment(self) -> bool:
        """Check if running in hybrid environment"""
        return self.ENVIRONMENT == "hybrid"