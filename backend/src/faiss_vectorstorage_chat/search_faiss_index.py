import faiss
from src.faiss_vectorstorage_chat.load_document import embedding_model, stored_documents
def search_faiss_index(query, index_name="vector_store", k=10):
    """
    Search the FAISS index for relevant document chunks.
    """
    # Load the FAISS index
    index = faiss.read_index(f"{index_name}.faiss")
    
    # Generate query embedding
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    # Perform search
    distances, indices = index.search(query_embedding, k)
    return indices, distances

