# rag_pipeline.py
# This file handles the RAG (Retrieval Augmented Generation) pipeline
# RAG = Before answering, AI searches a knowledge base for relevant info (like open-book exam)

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# ─────────────────────────────────────────────
# STEP 1: Load all knowledge base text files
# ─────────────────────────────────────────────
def load_knowledge_base(kb_folder="knowledge_base"):
    """
    Reads all .txt files from the knowledge_base folder
    Returns a list of Document objects
    """
    documents = []
    
    for filename in os.listdir(kb_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Each file becomes a Document with metadata
            doc = Document(
                page_content=content,
                metadata={"source": filename}
            )
            documents.append(doc)
    
    print(f"✅ Loaded {len(documents)} knowledge base files")
    return documents


# ─────────────────────────────────────────────
# STEP 2: Split documents into smaller chunks
# ─────────────────────────────────────────────
def split_documents(documents):
    """
    Splits large documents into smaller chunks for better retrieval
    chunk_size=500 means each chunk is ~500 characters
    chunk_overlap=50 means chunks share 50 characters at edges (for context)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# STEP 3: Create Vector Store (FAISS)
# ─────────────────────────────────────────────
def create_vector_store(chunks):
    """
    Converts text chunks into embeddings (numbers) and stores in FAISS
    FAISS = Fast library for finding similar vectors (similarity search)
    HuggingFace embeddings = FREE embedding model
    """
    print("⏳ Creating embeddings (this may take a moment)...")
    
    # Free HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create FAISS vector store from chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created!")
    
    return vector_store


# ─────────────────────────────────────────────
# STEP 4: Retrieve relevant chunks for a query
# ─────────────────────────────────────────────
def retrieve_context(vector_store, query, k=3):
    """
    Given a math question, finds the top-k most relevant chunks
    Returns the text content + source file names
    
    k=3 means return top 3 most relevant chunks
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    
    context_texts = []
    sources = []
    
    for doc, score in results:
        context_texts.append(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)
    
    combined_context = "\n\n---\n\n".join(context_texts)
    
    return combined_context, sources


# ─────────────────────────────────────────────
# MAIN: Initialize the RAG pipeline
# ─────────────────────────────────────────────
def initialize_rag(kb_folder="knowledge_base"):
    """
    Full pipeline: Load → Split → Embed → Store
    Returns the ready vector store
    """
    docs = load_knowledge_base(kb_folder)
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    return vector_store
