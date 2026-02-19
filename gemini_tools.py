"""
Enhanced Agentic Tools for Gemini API Function Calling
Defines tool schemas and execution functions for the RAG chatbot
"""

import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

# Tool schemas for Gemini API function calling
TOOL_SCHEMAS = [
    {
        "name": "retrieve_documents",
        "description": "Retrieve relevant documents from the wetland conservation knowledge base. Use this tool when you need to find information to answer the user's question. This searches across all available PDF documents.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "The search query to find relevant documents. Should be a clear, specific question or topic."
                },
                "top_k": {
                    "type": "INTEGER",
                    "description": "Number of top documents to retrieve (default: 8, max: 15)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_specific_document",
        "description": "Search for information within a specific document only. Use this when the user explicitly mentions a document name (e.g., 'National Wetland Policy', 'Metro Colombo Strategy') or asks to use only a specific source.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "document_name": {
                    "type": "STRING",
                    "description": "The name of the specific document to search within (e.g., 'National Wetland Policy.pdf')"
                },
                "query": {
                    "type": "STRING",
                    "description": "The search query within that specific document"
                },
                "top_k": {
                    "type": "INTEGER",
                    "description": "Number of top chunks to retrieve from this document (default: 5)"
                }
            },
            "required": ["document_name", "query"]
        }
    },
    {
        "name": "get_document_list",
        "description": "Get a list of all available documents in the knowledge base with their metadata. Use this when the user asks what documents are available or wants to know the sources.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
            "required": []
        }
    }
]


class ToolExecutor:
    """Executes tool calls for the agentic RAG system"""
    
    def __init__(self, rag_pipeline):
        """
        Initialize tool executor with RAG pipeline
        
        Args:
            rag_pipeline: RAGPipeline instance for document retrieval
        """
        self.rag_pipeline = rag_pipeline
        logger.info("ToolExecutor initialized with RAG pipeline")
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call and return results
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Dict with tool execution results
        """
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        try:
            if tool_name == "retrieve_documents":
                return self._retrieve_documents(tool_args)
            elif tool_name == "search_specific_document":
                return self._search_specific_document(tool_args)
            elif tool_name == "get_document_list":
                return self._get_document_list(tool_args)
            else:
                return {
                    "error": f"Unknown tool: {tool_name}",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _retrieve_documents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieve_documents tool"""
        # Ensure top_k is an integer
        try:
            top_k = int(args.get("top_k", 8))
        except (ValueError, TypeError):
            top_k = 8
        
        # Ensure query is a string
        query = str(args.get("query", args.get("question", "")))
        
        if not query:
            return {"error": "Query is required", "success": False}
        
        # Use the RAG pipeline's retrieval method
        if hasattr(self.rag_pipeline, 'hybrid_retriever'):
            retrieved_docs = self.rag_pipeline.hybrid_retriever.invoke(query)
            
            if not retrieved_docs:
                return {
                    "success": True,
                    "message": "No relevant documents found",
                    "documents": [],
                    "count": 0
                }
            
            # Apply relevance filtering
            top_docs = retrieved_docs[:top_k]
            filtered = self.rag_pipeline.relevance_checker.filter_documents(query, top_docs)
            filtered_docs = [d for d, s in filtered]
            
            # Format results
            results = []
            for doc in filtered_docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "type": doc.metadata.get("type", "text")
                })
            
            return {
                "success": True,
                "message": f"Retrieved {len(results)} relevant documents",
                "documents": results,
                "count": len(results)
            }
        else:
            return {
                "error": "RAG pipeline not initialized",
                "success": False
            }
    
    def _search_specific_document(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search_specific_document tool"""
        document_name = str(args.get("document_name", ""))
        query = str(args.get("query", ""))
        
        # Ensure top_k is an integer
        try:
            top_k = int(args.get("top_k", 5))
        except (ValueError, TypeError):
            top_k = 5
        
        if not document_name or not query:
            return {
                "error": "Both document_name and query are required",
                "success": False
            }
        
        # Retrieve all documents first
        if hasattr(self.rag_pipeline, 'hybrid_retriever'):
            retrieved_docs = self.rag_pipeline.hybrid_retriever.invoke(query)
            
            # Filter by specific document
            doc_specific = [
                doc for doc in retrieved_docs 
                if document_name.lower() in doc.metadata.get("source", "").lower()
            ]
            
            if not doc_specific:
                return {
                    "success": True,
                    "message": f"No content found in '{document_name}' for this query",
                    "documents": [],
                    "count": 0,
                    "searched_document": document_name
                }
            
            # Apply relevance filtering on document-specific results
            top_docs = doc_specific[:top_k * 2]  # Get more initially
            filtered = self.rag_pipeline.relevance_checker.filter_documents(query, top_docs)
            filtered_docs = [d for d, s in filtered[:top_k]]
            
            # Format results
            results = []
            for doc in filtered_docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "type": doc.metadata.get("type", "text")
                })
            
            return {
                "success": True,
                "message": f"Retrieved {len(results)} chunks from '{document_name}'",
                "documents": results,
                "count": len(results),
                "searched_document": document_name
            }
        else:
            return {
                "error": "RAG pipeline not initialized",
                "success": False
            }
    
    def _get_document_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_document_list tool"""
        if not hasattr(self.rag_pipeline, 'documents') or not self.rag_pipeline.documents:
            return {
                "success": False,
                "error": "No documents loaded in the knowledge base"
            }
        
        # Extract unique document names and metadata
        doc_info = {}
        for doc in self.rag_pipeline.documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in doc_info:
                doc_info[source] = {
                    "name": source,
                    "pages": set(),
                    "types": set(),
                    "chunk_count": 0
                }
            
            # Use metadata safely
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            doc_info[source]["pages"].add(str(metadata.get("page", "?")))
            doc_info[source]["types"].add(str(metadata.get("type", "text")))
            doc_info[source]["chunk_count"] += 1
        
        # Format for output
        documents = []
        for source, info in doc_info.items():
            documents.append({
                "name": info["name"],
                "total_chunks": info["chunk_count"],
                "page_count": len(info["pages"]),
                "content_types": list(info["types"])
            })
        
        return {
            "success": True,
            "message": f"Found {len(documents)} documents in knowledge base",
            "documents": documents,
            "total_documents": len(documents)
        }


def get_tool_schemas_for_gemini():
    """
    Convert tool schemas to Gemini API format
    
    Returns:
        List of tool declarations for Gemini API
    """
    return TOOL_SCHEMAS


def format_tool_result_for_prompt(tool_name: str, result: Dict[str, Any]) -> str:
    """
    Format tool execution result as a text prompt for the LLM
    
    Args:
        tool_name: Name of the executed tool
        result: Tool execution result
        
    Returns:
        Formatted string for inclusion in prompt
    """
    if not result.get("success", False):
        return f"[Tool Error - {tool_name}]: {result.get('error', 'Unknown error')}"
    
    if tool_name == "retrieve_documents" or tool_name == "search_specific_document":
        docs = result.get("documents", [])
        if not docs:
            return f"[{tool_name}]: No relevant documents found."
        
        formatted = f"[{tool_name}]: Retrieved {len(docs)} documents:\n\n"
        for i, doc in enumerate(docs, 1):
            formatted += f"--- Document {i} ---\n"
            formatted += f"Source: {doc['source']}, Page: {doc['page']}, Type: {doc['type']}\n"
            formatted += f"Content: {doc['content']}\n\n"
        
        return formatted
    
    elif tool_name == "get_document_list":
        docs = result.get("documents", [])
        formatted = f"[{tool_name}]: Available documents:\n\n"
        for doc in docs:
            formatted += f"- {doc['name']}: {doc['total_chunks']} chunks, {doc['page_count']} pages\n"
        
        return formatted
    
    return f"[{tool_name}]: {json.dumps(result, indent=2)}"
