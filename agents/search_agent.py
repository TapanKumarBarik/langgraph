# agents/search_agent.py
import json
from typing import Dict, Any, List
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from services.azure_services import AzureAISearchService
import structlog

logger = structlog.get_logger()

class SearchAgent:
    def __init__(self, llm: AzureChatOpenAI, search_service: AzureAISearchService):
        self.llm = llm
        self.search_service = search_service
        self.system_prompt = """You are a search specialist agent. Your role is to:

1. Analyze user queries to create effective search terms
2. Process search results to find relevant information
3. Extract key insights and prepare them for response generation

Guidelines:
- Create multiple search variations for complex queries
- Focus on the most relevant and recent information
- Note when search results contain images or visual content
- Identify authoritative sources and highlight them"""
    
    async def search_and_process(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Search for information and process results"""
        try:
            logger.info("Search agent processing query", query=query)
            
            # Generate search terms
            search_terms = await self._generate_search_terms(query, context)
            
            # Perform searches
            all_documents = []
            for term in search_terms[:3]:  # Limit to 3 search variations
                documents = await self.search_service.search_documents(
                    query=term,
                    top=5
                )
                all_documents.extend(documents)
            
            # Remove duplicates and rank
            unique_documents = self._deduplicate_documents(all_documents)
            ranked_documents = sorted(
                unique_documents, 
                key=lambda x: x.get("score", 0), 
                reverse=True
            )[:10]
            
            # Extract images and create citations
            images = self.search_service.extract_images_from_results(ranked_documents)
            citations = self.search_service.create_citations(ranked_documents)
            
            logger.info("Search completed", 
                       documents=len(ranked_documents),
                       images=len(images))
            
            return {
                "documents": ranked_documents,
                "images": images,
                "sources": citations,
                "search_terms_used": search_terms
            }
            
        except Exception as e:
            logger.error("Search processing failed", error=str(e))
            raise
    
    async def _generate_search_terms(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate multiple search term variations"""
        try:
            prompt = f"""Generate 2-3 effective search terms for this query: "{query}"

Context: {json.dumps(context, indent=2) if context else "None"}

Return search terms that would find the most relevant information. Consider:
- Key entities and concepts
- Alternative phrasings
- Related terminology

Format as a simple list, one term per line."""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse search terms
            terms = [line.strip() for line in response.content.split('\n') if line.strip()]
            
            # Always include the original query
            if query not in terms:
                terms.insert(0, query)
            
            return terms[:3]  # Maximum 3 terms
            
        except Exception as e:
            logger.error("Search term generation failed", error=str(e))
            return [query]  # Fallback to original query
    
    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on ID or content similarity"""
        seen_ids = set()
        unique_docs = []
        
        for doc in documents:
            doc_id = doc.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
            elif not doc_id:
                # For documents without ID, check content similarity
                content = doc.get("content", "")[:100]
                content_hash = hash(content)
                if content_hash not in seen_ids:
                    seen_ids.add(content_hash)
                    unique_docs.append(doc)
        
        return unique_docs