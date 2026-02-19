# Gemma 4B via Gemini API - Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### 3. Configure API Key

Create or update `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

Or set as environment variable:

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="your_api_key_here"

# Linux/Mac
export GOOGLE_API_KEY="your_api_key_here"
```

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

## What's New

### Enhanced Agentic RAG ✨

The chatbot now uses **function calling** for intelligent document retrieval:

1. **Smart Tool Selection**: The model automatically decides when to search documents
2. **Document-Specific Search**: Can search within a specific document when requested
3. **Better Context**: Uses only the most relevant information to answer questions
4. **Improved Accuracy**: Function calling reduces hallucinations

### Available Tools

- `retrieve_documents` - Search across all documents
- `search_specific_document` - Search within a named document
- `get_document_list` - List all available documents

### Example Queries

**General Query:**
> "What are the main wetland conservation strategies?"

The model will automatically call `retrieve_documents` to find relevant information.

**Document-Specific Query:**
> "What does the National Wetland Policy say about biodiversity?"

The model will call `search_specific_document` to search only that document.

## Model Information

- **Base Model**: Gemini 1.5 Flash
- **Grounding**: Gemma 4B capabilities
- **Context Window**: 32,768 tokens
- **Function Calling**: Enabled
- **Temperature**: 0.1 (for consistency)

## Troubleshooting

### "GOOGLE_API_KEY not found"
- Make sure you've set the API key in `.streamlit/secrets.toml`
- Or set it as an environment variable before running

### "Could not find import of google.generativeai"
- Run `pip install google-generativeai`
- Make sure you're in the correct virtual environment

### Function calling not working
- Check that you're using `gemini-1.5-flash` (not older models)
- Verify your API key has sufficient quota

## Architecture

```
User Query
    ↓
Gemini API (Gemma 4B)
    ↓
Function Call: retrieve_documents
    ↓
RAG Pipeline (Hybrid Retrieval + Reranking)
    ↓
Filtered Documents
    ↓
Gemini API (Answer Generation)
    ↓
Final Response with Citations
```

## Benefits Over Previous Setup

✅ **No more DeepSeek/HuggingFace rate limits**  
✅ **Better function calling support**  
✅ **Larger context window (32K vs previous)**  
✅ **More reliable API**  
✅ **Faster response times**  
✅ **Lower latency**
