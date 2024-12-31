from transformers import pipeline


def chat_with_documents(query, docs, max_context_length=1000):
    # Load an open-source LLM
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    
    # Create context by concatenating document contents
    context = "\n\n".join([doc.page_content[:max_context_length] for doc in docs])
    
    # Truncate context if it exceeds the maximum length
    if len(context) > max_context_length:
        context = context[:max_context_length]
    
    # Formulate the prompt
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate the response
    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]
