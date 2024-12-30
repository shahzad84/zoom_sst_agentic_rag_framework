from transformers import pipeline
import requests

# Load Hugging Face's free models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")  # Free QA model
text_generator = pipeline("text2text-generation", model="google/flan-t5-small")  # Better LLM model
 # Free text generation model


# Dummy retriever (replace with FAISS, Pinecone, etc.)
def retrieve_documents(query):
    """Mock retrieval for RAG. Replace with actual retriever."""
    documents = {
        "domain_query": "This is the relevant document content for domain queries.",
    }
    return documents.get(query, None)


# Web Search Logic
def decide_and_answer(query):
    """
    Decide whether to use RAG, Web Search, or LLM for answering.
    """
    # Check for domain-specific query
    document = retrieve_documents(query)
    if document:
        try:
            qa_result = qa_pipeline(question=query, context=document)
            return f"RAG Response: {qa_result['answer']}"
        except Exception as e:
            return f"Error using QA model: {str(e)}"

    # Default context for general knowledge QA
    default_context = """
    The Nile is the longest river in the world, stretching over 6,650 kilometers.
    The Amazon River is the second longest river, known for its vast watershed.
    Other major rivers include the Yangtze, Mississippi, and the Ganges.
    """

    # Check if the query is a question (simple heuristic)
    if "?" in query:
        try:
            qa_result = qa_pipeline(question=query, context=default_context)
            return f"QA Model Response: {qa_result['answer']}"
        except Exception as e:
            return f"Error using QA model: {str(e)}"

    # Use web search if domain-specific query not found
    if "search" in query.lower():
        search_results = decide_and_answer(query)
        return f"Web Search Results: {search_results}"

    # Fall back to LLM for open-ended queries
    try:
        # Use a more structured prompt for the model
        query_prompt = f"Answer the following question: {query}"
        answer = text_generator(query_prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
        return f"LLM Response: {answer}"
    except Exception as e:
        return f"Error using text generation model: {str(e)}"
