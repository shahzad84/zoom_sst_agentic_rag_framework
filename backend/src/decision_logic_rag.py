from transformers import pipeline
import requests

# Load Hugging Face's free models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")  # Free QA model
# text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")  # Updated text generation model
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")


# Dummy retriever (replace with FAISS, Pinecone, etc.)
def retrieve_documents(query):
    """Mock retrieval for RAG. Replace with actual retriever."""
    documents = {
        "domain_query": "This is the relevant document content for domain queries.",
    }
    return documents.get(query, None)

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
    
    # 1. Check for domain-specific queries (RAG)
    document = retrieve_documents(query)
    if document:
        try:
            print("Using RAG logic...")
            qa_result = qa_pipeline(question=query, context=document)
            if qa_result and qa_result["answer"]:
                return f"RAG Response: {qa_result['answer']}"
        except Exception as e:
            print(f"Error using QA model in RAG: {str(e)}")
    
    # 2. Use QA model for general knowledge questions
    # try:
    #     print("Using QA Model...")
    #     default_context = """
    #     The Nile is the longest river in the world, stretching over 6,650 kilometers.
    #     The Amazon River is the second longest river, known for its vast watershed.
    #     Other major rivers include the Yangtze, Mississippi, and the Ganges.
    #     """
    #     qa_result = qa_pipeline(question=query, context=default_context)
    #     if qa_result and qa_result["answer"]:
    #         return f"QA Model Response: {qa_result['answer']}"
    # except Exception as e:
    #     print(f"Error using QA model: {str(e)}")
    
    # 3. Perform Web Search as the next fallback
    try:
        print("Using Web Search...")
        web_result = web_search_duckduckgo(query)
        if web_result:
            return f"Web Search Results: {web_result}"
    except Exception as e:
        print(f"Error during web search: {str(e)}")
    
    # 4. Use LLM for open-ended queries as the last resort
    try:
        print("Using LLM for open-ended query...")
        query_prompt = f"Answer the following question: {query}"
        llm_result = text_generator(query_prompt, max_length=100, truncation=True, num_return_sequences=1)
        if llm_result and llm_result[0]["generated_text"]:
            return f"LLM Response: {llm_result[0]['generated_text']}"
    except Exception as e:
        print(f"Error using text generation model: {str(e)}")
    
    # 5. Fallback response if everything fails
    return "Sorry, I couldn't find an answer to your query."
