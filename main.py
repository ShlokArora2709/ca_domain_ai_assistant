# Folder PDF RAG System
# Process all PDFs from a local folder and create a RAG system

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
from typing import List, Dict
import re
from datetime import datetime

# Document processing
from sentence_transformers import SentenceTransformer
import faiss

# LLM Integration
import google.generativeai as genai

# PDF processing
import PyPDF2

class PDFProcessor:
    """Process all PDFs from a folder"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.metadata = []
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,;:()-]', '', text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)
                
        return chunks
    
    def process_pdf_file(self, file_path: str) -> List[Dict]:
        """Process a single PDF file"""
        try:
            filename = os.path.basename(file_path)
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += f"\n[Page {page_num + 1}]\n{page_text}\n"
                    except Exception as e:
                        st.warning(f"Error processing page {page_num + 1} of {filename}: {str(e)}")
                        continue
                
                if not text.strip():
                    st.warning(f"No text extracted from {filename}")
                    return []
                
                # Clean and chunk the text
                cleaned_text = self.clean_text(text)
                chunks = self.chunk_text(cleaned_text)
                
                processed_chunks = []
                for i, chunk in enumerate(chunks):
                    processed_chunks.append({
                        'content': chunk,
                        'source': filename,
                        'file_path': file_path,
                        'chunk_id': i,
                        'doc_type': 'PDF'
                    })
                
                return processed_chunks
                
        except Exception as e:
            st.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def process_folder(self, folder_path: str) -> List[Dict]:
        """Process all PDFs in a folder"""
        if not os.path.exists(folder_path):
            st.error(f"Folder not found: {folder_path}")
            return []
        
        # Find all PDF files
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        pdf_files.extend(glob.glob(os.path.join(folder_path, "*.PDF")))
        
        if not pdf_files:
            st.warning(f"No PDF files found in {folder_path}")
            return []
        
        st.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        all_chunks = []
        progress_bar = st.progress(0)
        
        for i, pdf_file in enumerate(pdf_files):
            st.text(f"Processing: {os.path.basename(pdf_file)}")
            progress_bar.progress((i + 1) / len(pdf_files))
            
            chunks = self.process_pdf_file(pdf_file)
            all_chunks.extend(chunks)
        
        progress_bar.empty()
        st.success(f"✅ Processed {len(all_chunks)} chunks from {len(pdf_files)} PDF files")
        return all_chunks

class RAGPipeline:
    """RAG pipeline with vector search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_store = None
        self.documents = []
        self.metadata = []
        self.index = None
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        return self.embedding_model.encode(texts, show_progress_bar=True)
    
    def build_vector_store(self, documents: List[Dict]):
        """Build FAISS vector store from documents"""
        self.documents = documents
        self.metadata = [doc for doc in documents]
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        if not texts:
            st.error("No text content found to build vector store")
            return
        
        # Generate embeddings
        st.info("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        st.success(f"✅ Vector store built with {len(documents)} document chunks")
        
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant document chunks for a query"""
        if self.index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save_index(self, filepath: str):
        """Save the vector index and metadata"""
        if self.index is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            faiss.write_index(self.index, f"{filepath}.index")
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'documents': self.documents
                }, f)
            st.success(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load the vector index and metadata"""
        try:
            self.index = faiss.read_index(f"{filepath}.index")
            with open(f"{filepath}.metadata", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.documents = data['documents']
            st.success(f"Index loaded from {filepath}")
            return True
        except Exception as e:
            st.error(f"Failed to load index: {str(e)}")
            return False

class DocumentAssistant:
    """AI assistant for document queries using Gemini LLM"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-002')
        self.rag_pipeline = RAGPipeline()
        
    def create_context_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Create prompt with retrieved context"""
        
        context = "\n\n".join([
            f"Document: {chunk['source']}\nContent: {chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""You are an AI assistant that answers questions based on the provided document context.

Context Documents:
{context}

User Query: {query}

Instructions:
1. Answer the query based ONLY on the information provided in the context documents
2. If the answer is not in the context, clearly state that the information is not available
3. Provide accurate and detailed responses
4. Include relevant quotes from the documents when helpful
5. Always cite which document the information comes from
6. If multiple documents contain relevant information, synthesize the information appropriately

Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, max_tokens: int = 1000) -> Dict:
        """Generate response using RAG + Gemini"""
        
        # Retrieve relevant context
        relevant_chunks = self.rag_pipeline.retrieve_relevant_chunks(query, top_k=5)
        
        if not relevant_chunks:
            return {
                'answer': "I don't have sufficient information in my knowledge base to answer this query. Please ensure the PDF documents have been processed.",
                'sources': [],
                'confidence': 0.0,
                'relevant_chunks': []
            }
        
        # Create prompt with context
        prompt = self.create_context_prompt(query, relevant_chunks)
        
        try:
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Extract sources
            sources = list(set([chunk['source'] for chunk in relevant_chunks]))
            avg_confidence = np.mean([chunk['similarity_score'] for chunk in relevant_chunks])
            
            return {
                'answer': response.text,
                'sources': sources,
                'confidence': float(avg_confidence),
                'relevant_chunks': relevant_chunks
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'relevant_chunks': []
            }

def main():
    st.set_page_config(
        page_title="CA assistant",
        page_icon="",
        layout="wide"
    )
    
    st.title(" CA Document Assistant")
    # st.subtitle("Ask questions about your PDF documents")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'folder_path' not in st.session_state:
        st.session_state.folder_path = "./pdfs"
    
    # Sidebar configuration
    with st.sidebar:
        st.header(" Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if api_key and not st.session_state.assistant:
            st.session_state.assistant = DocumentAssistant(api_key)
            st.success(" Assistant initialized!")
        
        # st.header("PDF Folder Processing")
        
        # # Folder path input
        # folder_path = st.text_input(
        #     "PDF Folder Path",
        #     value="./pdfs",
        #     help="Enter the path to your PDF folder (e.g., ./pdfs, C:/documents/pdfs)"
        # )
        folder_path = "./pdfs"
        # st.session_state.folder_path = folder_path
        
        # Process folder button
        if st.button("Process PDF Folder", disabled=not api_key):
            if not api_key:
                st.error("Please provide API key first!")
            elif not folder_path.strip():
                st.error("Please provide folder path!")
            else:
                process_pdf_folder(folder_path)
        
        # Save/Load index
        st.header("Index Management")
        
        index_name = st.text_input("Index Name", value="pdf_index")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Index"):
                if st.session_state.assistant and st.session_state.documents_processed:
                    st.session_state.assistant.rag_pipeline.save_index(f"./indexes/{index_name}")
        
        with col2:
            if st.button("Load Index"):
                if st.session_state.assistant:
                    if st.session_state.assistant.rag_pipeline.load_index(f"./indexes/{index_name}"):
                        st.session_state.documents_processed = True
        
        # Document stats
        if st.session_state.documents_processed and st.session_state.assistant:
            if hasattr(st.session_state.assistant.rag_pipeline, 'metadata'):
                total_chunks = len(st.session_state.assistant.rag_pipeline.metadata)
                unique_sources = len(set([doc['source'] for doc in st.session_state.assistant.rag_pipeline.metadata]))
                st.success(f"{total_chunks} chunks from {unique_sources} PDFs")
        
        with st.expander(" Document Overview"):
            if st.session_state.documents_processed and st.session_state.assistant:
                metadata = st.session_state.assistant.rag_pipeline.metadata
                if metadata:
                    # Create summary statistics
                    sources = [doc['source'] for doc in metadata]
                    source_counts = pd.Series(sources).value_counts()
                    
                    st.subheader("Document Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Chunks", len(metadata))
                    with col2:
                        st.metric("Total Documents", len(source_counts))
                    with col3:
                        st.metric("Avg Chunks per Doc", f"{len(metadata) / len(source_counts):.1f}")
                    
                    st.subheader("Documents by Chunk Count")
                    st.bar_chart(source_counts)
    
    # Main interface
    if not api_key:
        st.warning(" Please provide your Gemini API key in the sidebar to get started.")
        st.info("""
        **How to get a Gemini API key:**
        1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Copy and paste it in the sidebar
        """)
        return
    
    if not st.session_state.documents_processed:
        st.info("Please process your PDF folder to start querying.")
        
        # Show folder info
        if folder_path and os.path.exists(folder_path):
            pdf_files = glob.glob(os.path.join(folder_path, "*.pdf")) + glob.glob(os.path.join(folder_path, "*.PDF"))
            if pdf_files:
                st.write(f"**Found {len(pdf_files)} PDF files in '{folder_path}':**")
                for pdf_file in pdf_files[:10]:  # Show first 10 files
                    st.write(f"• {os.path.basename(pdf_file)}")
                if len(pdf_files) > 10:
                    st.write(f"... and {len(pdf_files) - 10} more files")
            else:
                st.warning(f"No PDF files found in '{folder_path}'")
        elif folder_path:
            st.error(f"Folder not found: '{folder_path}'")
        
        return
    
    # Chat interface
    st.header(" Ask Questions About Your Documents")

    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['query'])
        with st.chat_message("assistant"):
            st.write(chat['response']['answer'])
            
            # Show sources and confidence
            col1, col2 = st.columns([3, 1])
            with col1:
                if chat['response']['sources']:
                    st.caption(f" Sources: {', '.join(chat['response']['sources'])}")
            with col2:
                confidence = chat['response']['confidence']
                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                st.caption(f" :{confidence_color}[{confidence:.2f}]")
    
    # Query input
    user_query = st.chat_input("Ask me anything about your PDF documents...")
    
    if user_query:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching through your documents..."):
                response = st.session_state.assistant.generate_response(user_query)
            
            st.write(response['answer'])
            
            # Show sources and confidence
            col1, col2 = st.columns([3, 1])
            with col1:
                if response['sources']:
                    st.caption(f" Sources: {', '.join(response['sources'])}")
            with col2:
                confidence = response['confidence']
                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                st.caption(f" :{confidence_color}[{confidence:.2f}]")

        # Add to chat history
        st.session_state.chat_history.append({
            'query': user_query,
            'response': response,
            'timestamp': datetime.now()
        })
    
    # Advanced search
    with st.expander(" Advanced Document Search"):
        if st.session_state.documents_processed:
            search_query = st.text_input("Search in documents:", key="search_input")
            num_results = st.slider("Number of results:", 1, 20, 10)
            
            if search_query:
                chunks = st.session_state.assistant.rag_pipeline.retrieve_relevant_chunks(
                    search_query, top_k=num_results
                )
                
                st.subheader(f"Search Results for: '{search_query}'")
                for i, chunk in enumerate(chunks):
                    with st.container():
                        st.write(f"**Result {i+1}** - Score: {chunk['similarity_score']:.3f}")
                        st.write(f"**Source:** {chunk['source']}")
                        st.write(f"**Content:** {chunk['content'][:500]}{'...' if len(chunk['content']) > 500 else ''}")
                        st.divider()
    
    # Document overview
    

def process_pdf_folder(folder_path: str):
    """Process all PDFs in the specified folder"""
    if not st.session_state.assistant:
        st.error("Assistant not initialized!")
        return
    
    processor = PDFProcessor()
    
    with st.spinner("Processing PDF files..."):
        all_chunks = processor.process_folder(folder_path)
    
    if all_chunks:
        with st.spinner("Building vector index..."):
            st.session_state.assistant.rag_pipeline.build_vector_store(all_chunks)
        st.session_state.documents_processed = True
    else:
        st.error("No documents could be processed from the folder")

if __name__ == "__main__":
    main()