# agents/sql_agent.py
import json
from typing import Dict, Any, List
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from services.azure_services import AzureSQLService
import structlog

logger = structlog.get_logger()

class SQLAgent:
    def __init__(self, llm: AzureChatOpenAI, sql_service: AzureSQLService):
        self.llm = llm
        self.sql_service = sql_service
        self.system_prompt = """You are a SQL specialist agent. Your role is to:

1. Convert natural language queries into safe SQL queries
2. Execute queries against the database
3. Process and analyze the results

Available tables and their columns:
{}

Guidelines:
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
- Always use TOP clause to limit results (max 50 rows)
- Use appropriate WHERE clauses for filtering
- Consider JOINs when data spans multiple tables
- Handle date ranges and time-based queries appropriately
- Provide meaningful column aliases for complex calculations

Safety rules:
- Never execute queries that could modify data
- Always validate input parameters
- Use parameterized queries when possible""".format(
            json.dumps(sql_service.get_all_tables_info() if sql_service else [], indent=2)
        )
    
    async def query_and_process(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert query to SQL and execute"""
        try:
            logger.info("SQL agent processing query", query=query)
            
            # Generate SQL query
            sql_query = await self._generate_sql_query(query, context)
            
            # Execute query
            results = await self.sql_service.execute_query(sql_query)
            
            # Create sources/citations
            sources = [{
                "id": "sql_query_result",
                "title": "Database Query Result",
                "source": "Azure SQL Database",
                "query": sql_query,
                "result_count": len(results)
            }]
            
            logger.info("SQL query completed", 
                       results_count=len(results),
                       query_preview=sql_query[:100])
            
            return {
                "data": results,
                "sql_query": sql_query,
                "sources": sources
            }
            
        except Exception as e:
            logger.error("SQL processing failed", error=str(e))
            raise
    
    async def _generate_sql_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> str:
        """Generate SQL query from natural language"""
        try:
            table_info = self.sql_service.get_all_tables_info()
            
            prompt = f"""Convert this natural language query to SQL: "{query}"

Available tables and columns:
{json.dumps(table_info, indent=2)}

Context: {json.dumps(context, indent=2) if context else "None"}

Generate a safe SELECT query that:
1. Uses appropriate tables and JOINs
2. Includes TOP 50 to limit results
3. Has meaningful WHERE clauses
4. Uses proper column names and aliases

Return only the SQL query, no explanations."""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Clean up the SQL query
            sql_query = response.content.strip()
            
            # Remove markdown formatting if present
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            sql_query = sql_query.strip()
            
            return sql_query
            
        except Exception as e:
            logger.error("SQL generation failed", error=str(e))
            raise
