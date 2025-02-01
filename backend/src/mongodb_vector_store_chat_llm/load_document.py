from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(file_path: str):
    """Load and split PDF document into chunks"""
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    pages = loader.load()
    return text_splitter.split_documents(pages)


def summarize_context(context: str, max_length: int = 1000) -> str:
    """Summarize context to fit within max_length"""
    if len(context) <= max_length:
        return context
    
    # Split into sentences and take the most relevant ones
    sentences = context.split(". ")
    summarized = ". ".join(sentences[:10])  # Adjust as needed
    return summarized[:max_length]