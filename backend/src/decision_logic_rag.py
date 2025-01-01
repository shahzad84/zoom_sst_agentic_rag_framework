from transformers import pipeline
import requests
from src.faiss_vectorstorage_chat.search_faiss_index import search_faiss_index
from src.faiss_vectorstorage_chat.chat_with_document import chat_with_documents
from src.faiss_vectorstorage_chat.load_document import stored_documents


# Load Hugging Face's free models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")  # Free QA model
# text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")  # Updated text generation model
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")



def web_search_duckduckgo(query):
    """
    Perform a web search using the DuckDuckGo API.
    """
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        # print("DuckDuckGo API Response:", data)

        # Check the response content
        if data.get("Abstract"):
            return data["Abstract"]
        elif data.get("RelatedTopics"):
            # Fetch the first related topic if abstract is empty
            first_topic = data["RelatedTopics"][0]
            return first_topic["Text"] if "Text" in first_topic else "No text in related topics."
        else:
            return None
    except requests.exceptions.RequestException as e:
        return f"Error during web search request: {str(e)}"
    except Exception as e:
        return f"Error processing web search response: {str(e)}"

# print(web_search_duckduckgo("What is the ozone layer?"))
 
def decide_and_answer(query):
    """
    Decide whether to use RAG, QA, Web Search, or LLM for answering.
    """
    print(f"Deciding handler for query: {query}")
    
    # 1. Use FAISS for document retrieval
    try:
        print("Using RAG...")
        indices, distances = search_faiss_index(query, index_name="vector_store", k=10)
        retrieved_docs = [stored_documents[i] for i in indices[0] if i < len(stored_documents)]
        
        if retrieved_docs:
            # print(f"Retrieved Docs: {[doc.page_content for doc in retrieved_docs]}")
            response = chat_with_documents(query, retrieved_docs)
            
            # Refine the response
            if response and response.strip().lower() != query.lower() and len(response.split()) > 3:

                print(f"RAG Response: {response}")
                return {"method": "RAG", "response": response}
            
    except Exception as e:
        print(f"Error during FAISS query or document chat: {str(e)}")
    
    # 2. Perform Web Search as the next fallback
    try:
        print("Using Web Search...")
        web_result = web_search_duckduckgo(query)
        if web_result:
            return f"Web Search Results: {web_result}"
    except Exception as e:
        print(f"Error during web search: {str(e)}")
    
    # 3. Use LLM for open-ended queries as the last resort
    try:
        print("Using LLM for open-ended query...")
        query_prompt = f"Answer the following question: {query}"
        llm_result = text_generator(query_prompt, max_length=100, truncation=True, num_return_sequences=1)
        if llm_result and llm_result[0]["generated_text"]:
            return f"LLM Response: {llm_result[0]['generated_text']}"
    except Exception as e:
        print(f"Error using text generation model: {str(e)}")
    
    # 4. Fallback response if everything fails
    return "Sorry, I couldn't find an answer to your query."
