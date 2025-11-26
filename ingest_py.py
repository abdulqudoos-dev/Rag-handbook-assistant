"""
FYP Handbook RAG - Ingestion Pipeline
Extracts text from PDF, chunks it, creates embeddings, and builds FAISS index
"""

import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from typing import List, Dict
import os

# Configuration
CHUNK_SIZE = 300  # words
OVERLAP = 0.3  # 30% overlap
MODEL_NAME = 'all-MiniLM-L6-v2'
PDF_PATH = 'FYP-Handbook-2023.pdf'
INDEX_PATH = 'faiss_index.bin'
CHUNKS_PATH = 'chunks_metadata.pkl'

class DocumentChunker:
    def __init__(self, chunk_size=300, overlap=0.3):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_first_heading(self, text):
        """Extract first heading-like text (usually uppercase or title case)"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and (line.isupper() or len(line.split()) <= 8):
                return line
        return "Unknown Section"
    
    def chunk_text(self, text, page_num):
        """Split text into overlapping chunks with metadata"""
        words = text.split()
        chunks = []
        
        section_hint = self.extract_first_heading(text)
        
        step_size = int(self.chunk_size * (1 - self.overlap))
        
        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:  # Skip very small chunks
                continue
            
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'page': page_num,
                'section_hint': section_hint,
                'chunk_id': f"p{page_num}_c{len(chunks)}"
            })
        
        return chunks

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF preserving page numbers"""
    print(f"Loading PDF: {pdf_path}")
    pages_data = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        print(f"Total pages: {total_pages}")
        
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            if text.strip():
                pages_data.append({
                    'page_num': page_num + 1,
                    'text': text
                })
    
    print(f"Extracted text from {len(pages_data)} pages")
    return pages_data

def create_chunks(pages_data):
    """Create chunks from all pages"""
    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    all_chunks = []
    
    for page_data in pages_data:
        page_chunks = chunker.chunk_text(page_data['text'], page_data['page_num'])
        all_chunks.extend(page_chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    return all_chunks

def create_embeddings(chunks):
    """Create embeddings using Sentence-BERT"""
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    texts = [chunk['text'] for chunk in chunks]
    print(f"Creating embeddings for {len(texts)} chunks...")
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings, model

def build_faiss_index(embeddings):
    """Build FAISS index for similarity search"""
    dimension = embeddings.shape[1]
    print(f"Building FAISS index with dimension: {dimension}")
    
    # Using L2 distance (can convert to cosine similarity)
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    index.add(embeddings.astype('float32'))
    print(f"Added {index.ntotal} vectors to index")
    
    return index

def save_index_and_metadata(index, chunks):
    """Save FAISS index and chunk metadata to disk"""
    print(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"Saving chunks metadata to {CHUNKS_PATH}")
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    
    print("✓ Index and metadata saved successfully")

def main():
    """Main ingestion pipeline"""
    print("=" * 60)
    print("FYP Handbook RAG - Ingestion Pipeline")
    print("=" * 60)
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF file not found: {PDF_PATH}")
        print("Please place the FYP-Handbook-2023.pdf in the current directory")
        return
    
    # Step 1: Extract text from PDF
    pages_data = extract_text_from_pdf(PDF_PATH)
    
    # Step 2: Create chunks
    chunks = create_chunks(pages_data)
    
    # Step 3: Create embeddings
    embeddings, model = create_embeddings(chunks)
    
    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Step 5: Save to disk
    save_index_and_metadata(index, chunks)
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Chunk size: ~{CHUNK_SIZE} words with {int(OVERLAP*100)}% overlap")
    print(f"\nFiles created:")
    print(f"  - {INDEX_PATH}")
    print(f"  - {CHUNKS_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
