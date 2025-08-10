# config/settings.py

import os
from pydantic import BaseSettings, validator
from typing import Optional, Literal

class Settings(BaseSettings):
    # Environment Configuration
    ENVIRONMENT: Literal["local", "azure", "hybrid"] = os.getenv("ENVIRONMENT", "azure")
    
    # Azure OpenAI (can be local or Azure)
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "http://localhost:1234/v1")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "local-key")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    OPENAI_API_TYPE: Literal["azure", "openai"] = os.getenv("OPENAI_API_TYPE", "azure")
    
    # Storage Configuration (Azure Blob or Local File System)
    USE_LOCAL_STORAGE: bool = os.getenv("USE_LOCAL_STORAGE", "false").lower() == "true"
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "./local_storage/conversations")
    CONVERSATION_CONTAINER: str = os.getenv("CONVERSATION_CONTAINER", "conversations")
    
    # Search Configuration (Azure AI Search or Local)
    USE_LOCAL_SEARCH: bool = os.getenv("USE_LOCAL_SEARCH", "false").lower() == "true"
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "knowledge-base")
    LOCAL_SEARCH_INDEX_PATH: str = os.getenv("LOCAL_SEARCH_INDEX_PATH", "./local_storage/search_index")
    
    # SQL Configuration (Azure SQL or Local SQL Server Express)
    USE_LOCAL_SQL: bool = os.getenv("USE_LOCAL_SQL", "false").lower() == "true"
    AZURE_SQL_CONNECTION_STRING: str = os.getenv("AZURE_SQL_CONNECTION_STRING", "")
    LOCAL_SQL_SERVER: str = os.getenv("LOCAL_SQL_SERVER", "localhost\\SQLEXPRESS")
    LOCAL_SQL_DATABASE: str = os.getenv("LOCAL_SQL_DATABASE", "AdventureWorks2022")
    LOCAL_SQL_USERNAME: str = os.getenv("LOCAL_SQL_USERNAME", "")
    LOCAL_SQL_PASSWORD: str = os.getenv("LOCAL_SQL_PASSWORD", "")
    LOCAL_SQL_TRUSTED_CONNECTION: bool = os.getenv("LOCAL_SQL_TRUSTED_CONNECTION", "true").lower() == "true"
    
    # Redis (optional for caching)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    USE_REDIS: bool = os.getenv("USE_REDIS", "false").lower() == "true"
    
    # API Configuration
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    MAX_SQL_RESULTS: int = int(os.getenv("MAX_SQL_RESULTS", "50"))
    
    # Feature Flags
    ENABLE_IMAGE_SUPPORT: bool = os.getenv("ENABLE_IMAGE_SUPPORT", "true").lower() == "true"
    ENABLE_FOLLOWUP_SUGGESTIONS: bool = os.getenv("ENABLE_FOLLOWUP_SUGGESTIONS", "true").lower() == "true"
    
    @validator('LOCAL_SQL_CONNECTION_STRING', always=True)
    def build_local_sql_connection_string(cls, v, values):
        """Build local SQL Server connection string"""
        if values.get('USE_LOCAL_SQL'):
            server = values.get('LOCAL_SQL_SERVER', 'localhost\\SQLEXPRESS')
            database = values.get('LOCAL_SQL_DATABASE', 'AdventureWorks2022')
            username = values.get('LOCAL_SQL_USERNAME')
            password = values.get('LOCAL_SQL_PASSWORD')
            trusted = values.get('LOCAL_SQL_TRUSTED_CONNECTION', True)
            
            if trusted:
                return f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes"
            else:
                return f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        return v
    
    LOCAL_SQL_CONNECTION_STRING: str = ""
    
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
    
    class Config:
        env_file = ".env"