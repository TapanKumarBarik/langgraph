# agents/followup_agent.py
import json
from typing import Dict, Any, List
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import structlog

logger = structlog.get_logger()

class FollowupAgent:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are a followup specialist agent. Your role is to generate relevant follow-up questions and suggestions based on the user's query and the results retrieved.

Guidelines:
- Generate 2-4 relevant follow-up questions
- Consider what additional information the user might want
- Base suggestions on the actual results found
- Make questions specific and actionable
- Avoid repetitive or obvious questions

Focus on:
- Deeper analysis of the results
- Related topics or data
- Clarifying questions
- Next steps or actions"""
    
    async def generate_followups(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        sql_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up suggestions"""
        try:
            # Prepare result summary
            result_summary = self._create_result_summary(search_results, sql_results)
            
            prompt = f"""User query: "{query}"

Results summary:
{result_summary}

Context: {json.dumps(context, indent=2) if context else "None"}

Generate 3 relevant follow-up questions the user might want to ask next. Base them on the actual results found.

Return as a simple list, one question per line."""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse follow-up questions
            followups = [
                line.strip().lstrip('- ').lstrip('* ').lstrip('1. ').lstrip('2. ').lstrip('3. ')
                for line in response.content.split('\n') 
                if line.strip() and '?' in line
            ]
            
            return followups[:4]  # Maximum 4 followups
            
        except Exception as e:
            logger.error("Followup generation failed", error=str(e))
            return []
    
    def _create_result_summary(
        self, 
        search_results: List[Dict[str, Any]], 
        sql_results: List[Dict[str, Any]]
    ) -> str:
        """Create a summary of results for context"""
        summary_parts = []
        
        if search_results:
            summary_parts.append(f"Found {len(search_results)} search results covering topics like: " + 
                               ", ".join([doc.get("title", "")[:50] for doc in search_results[:3]]))
        
        if sql_results:
            summary_parts.append(f"Database query returned {len(sql_results)} records")
            if sql_results:
                # Get column names
                columns = list(sql_results[0].keys()) if sql_results else []
                summary_parts.append(f"Data includes columns: {', '.join(columns[:5])}")
        
        return ". ".join(summary_parts) if summary_parts else "No specific results found"
