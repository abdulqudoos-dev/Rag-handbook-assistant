"""
FYP Handbook RAG - Streamlit Interface
Query the handbook and get grounded answers with page citations
"""

import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_PATH = 'faiss_index.bin'
CHUNKS_PATH = 'chunks_metadata.pkl'
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25

# System prompt template
SYSTEM_PROMPT = """You are a helpful FYP Handbook assistant for FAST-NUCES students. 

IMPORTANT RULES:
1. Answer ONLY using information from the provided context chunks
2. ALWAYS cite page numbers using the format (p. X) after claims
3. If the context doesn't contain the answer, say "I don't have that information in the handbook"
4. Do NOT make up information or use external knowledge
5. Be concise but complete in your answers
6. Quote key phrases when helpful, but also paraphrase to explain clearly

Question: {question}

Context from FYP Handbook:
{context}

Answer (with page citations):"""

@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_index_and_chunks():
    """Load FAISS index and chunk metadata"""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None
    
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    
    return index, chunks

def retrieve_chunks(query, model, index, chunks, top_k=TOP_K):
    """Retrieve top-k most similar chunks"""
    # Encode query
    query_embedding = model.encode([query])
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    query_embedding = -query_embedding 
    # Search
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    
    # Convert L2 distances to cosine similarities
    # After normalization, L2 distance d relates to cosine similarity: sim = 1 - d²/2
    similarities = 1 - (distances[0] ** 2) / 2
    
    # Get retrieved chunks
    retrieved = []
    for idx, sim in zip(indices[0], similarities):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk['similarity'] = float(sim)
            retrieved.append(chunk)
    
    return retrieved

def format_context(chunks):
    """Format retrieved chunks into context string"""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Chunk {i} - Page {chunk['page']}]\n{chunk['text']}\n"
        )
    return "\n".join(context_parts)

def generate_answer(question, context):
    """
    Generate a simple grounded answer directly from the retrieved context.

    The implementation stays lightweight by:
    1. Parsing the chunk blocks that include page numbers.
    2. Extracting the first meaningful sentence from each chunk.
    3. Returning a bulleted summary that cites the relevant page.
    """
    if not context or not context.strip():
        return "I don't have that information in the handbook."

    # Pattern captures "[Chunk X - Page Y]" headers plus their text
    pattern = re.compile(
        r"\[Chunk\s+\d+\s+-\s+Page\s+(\d+)\]\s*(.*?)(?=\n\[Chunk|\Z)",
        flags=re.DOTALL
    )
    matches = pattern.findall(context)

    if not matches:
        return "I don't have that information in the handbook."

    summary_lines = ["Here is what the handbook states:"]

    for page, chunk_text in matches:
        cleaned = chunk_text.strip()
        if not cleaned:
            continue

        # Split into basic sentences; keep the first informative one
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        first_sentence = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 20:  # avoid fragments
                first_sentence = sentence
                break

        if not first_sentence:
            first_sentence = cleaned[:200].strip()

        if first_sentence:
            summary_lines.append(f"- {first_sentence} (p. {page})")

    if len(summary_lines) == 1:
        return "I don't have that information in the handbook."

    return "\n".join(summary_lines)

def check_relevance(chunks, threshold=SIMILARITY_THRESHOLD):
    """Check if any retrieved chunks meet the similarity threshold"""
    if not chunks:
        return False
    max_sim = max(chunk['similarity'] for chunk in chunks)
    return max_sim >= threshold

def generate_answer_with_llm(question, context):
    """Generate answer using OpenAI API"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = SYSTEM_PROMPT.format(
            question=question,
            context=context
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" for better quality
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for factual answers
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="FYP Handbook Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 FAST-NUCES FYP Handbook Assistant")
    st.markdown("Ask questions about the Final Year Project process and get answers grounded in the official handbook.")
    
    # Load resources
    with st.spinner("Loading model and index..."):
        model = load_model()
        index, chunks = load_index_and_chunks()
    
    if index is None or chunks is None:
        st.error("❌ Index not found! Please run `ingest.py` first to process the handbook.")
        st.info("Run: `python ingest.py` in your terminal")
        return
    
    st.success(f"✓ Loaded {len(chunks)} chunks from handbook")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Top K Chunks", 3, 10, (TOP_K))
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, SIMILARITY_THRESHOLD, 0.05)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This RAG system uses:
        - **Model**: all-MiniLM-L6-v2
        - **Vector Store**: FAISS
        - **Chunk Size**: ~300 words
        - **Overlap**: 30%
        """)
    
    # Main interface
    st.markdown("---")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What headings, fonts, and sizes are required in the FYP report?
        - What margins and spacing do we use?
        - What are the required chapters/sections of a Development FYP report?
        - What are the required chapters of an R&D-based FYP report?
        - How should endnotes like 'Ibid.' and 'op. cit.' be used?
        - What goes into the Executive Summary and Abstract?
        """)
    
    # Query input
    user_question = st.text_area(
        "Your Question:",
        placeholder="e.g., What are the formatting requirements for the FYP report?",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if ask_button and user_question.strip():
        with st.spinner("Searching handbook..."):
            # Retrieve relevant chunks
            retrieved_chunks = retrieve_chunks(
                user_question, 
                model, 
                index, 
                chunks, 
                top_k=top_k
            )
            
            # Check relevance
            if not check_relevance(retrieved_chunks, threshold):
                st.warning("⚠️ I don't have that information in the handbook. The retrieved content has low similarity to your question.")
                st.info("Try rephrasing your question or asking about topics covered in the FYP handbook.")
            else:
                # Format context
                context = format_context(retrieved_chunks)
                
                # Generate prompt (in a full system, this would call an LLM)
                prompt = generate_answer(user_question, context)
                
                # Display answer section
                st.markdown("### 📝 Answer")
                
                # Generate answer with OpenAI
                with st.spinner("Generating answer..."):
                    answer = generate_answer_with_llm(user_question, context)
                
                # Display the answer
                st.markdown(answer)
                
                # Optional: Show the prompt used
                with st.expander("🔍 View Prompt & Context Sent to LLM"):
                    prompt = SYSTEM_PROMPT.format(
                        question=user_question,
                        context=context
                    )
                    st.code(prompt, language="text")
                
                # Show retrieved chunks
                st.markdown("### 📄 Retrieved Sources")
                
                for i, chunk in enumerate(retrieved_chunks, 1):
                    with st.expander(
                        f"Source {i} - Page {chunk['page']} "
                        f"(Similarity: {chunk['similarity']:.3f}) - {chunk['section_hint'][:50]}..."
                    ):
                        st.markdown(f"**Page:** {chunk['page']}")
                        st.markdown(f"**Section:** {chunk['section_hint']}")
                        st.markdown(f"**Similarity Score:** {chunk['similarity']:.4f}")
                        st.markdown("**Content:**")
                        st.text_area("", chunk['text'], height=200, key=f"chunk_{i}")
                
                # Summary of sources
                pages = sorted(set(chunk['page'] for chunk in retrieved_chunks))
                st.info(f"📌 Information found on pages: {', '.join(map(str, pages))}")

if __name__ == "__main__":
    main()
