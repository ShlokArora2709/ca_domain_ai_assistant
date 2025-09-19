# CA Domain AI Assistant

A Retrieval Augmented Generation (RAG) system built with Streamlit for querying PDF documents in the Chartered Accountancy domain. The application processes PDF documents, creates vector embeddings, and provides intelligent question-answering capabilities using Google's Gemini AI model.

## Features

- **PDF Document Processing**: Automatically processes multiple PDF files from a local folder
- **Vector Search**: Uses FAISS for efficient similarity search across document chunks
- **AI-Powered Responses**: Leverages Google Gemini for generating contextual answers
- **Interactive Chat Interface**: Streamlit-based web interface for seamless user interaction
- **Document Overview**: Statistics and insights about processed documents
- **Advanced Search**: Customizable search with adjustable result limits
- **Persistent Storage**: Saves and loads vector indexes for faster startup

## [Live Demo](https://shlokarora2709-ca-domain-ai-assistant-main-nxzpfw.streamlit.app/)


## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- PDF documents for processing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ShlokArora2709/ca_domain_ai_assistant.git
cd ca_domain_ai_assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory and add your Gemini API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Place your PDF documents in the `pdfs/` folder (or update the folder path in the code)

### Running the Application

```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## Code Architecture

### Core Components

#### 1. PDFProcessor Class
- **Purpose**: Handles PDF document processing and text extraction
- **Key Methods**:
  - `process_pdf_file()`: Extracts text from individual PDF files
  - `process_folder()`: Batch processes all PDFs in a folder
  - `chunk_text()`: Splits documents into overlapping chunks for better retrieval
  - `clean_text()`: Preprocesses text by removing noise and formatting

#### 2. RAGPipeline Class
- **Purpose**: Manages vector embeddings and similarity search
- **Key Methods**:
  - `build_vector_store()`: Creates FAISS index from document embeddings
  - `retrieve_relevant_chunks()`: Finds most similar document chunks for queries
  - `save_index()`/`load_index()`: Persists vector index for faster loading

#### 3. DocumentAssistant Class
- **Purpose**: Integrates RAG with Google Gemini for intelligent responses
- **Key Methods**:
  - `generate_response()`: Combines retrieved context with LLM generation
  - `create_context_prompt()`: Formats retrieved documents into effective prompts

### Technical Stack

- **Frontend**: Streamlit for web interface
- **Vector Database**: FAISS for similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 1.5 Flash
- **PDF Processing**: PyPDF2
- **Data Processing**: NumPy, Pandas

### Data Flow

1. **Document Ingestion**: PDFs are processed and split into chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Index Creation**: FAISS index is built for efficient similarity search
4. **Query Processing**: User queries are embedded and matched against document chunks
5. **Context Retrieval**: Most relevant chunks are retrieved based on similarity scores
6. **Response Generation**: Retrieved context is combined with user query and sent to Gemini
7. **Answer Delivery**: AI-generated response is displayed with source attribution

## Usage Guide

### Initial Setup

1. Start the application using `streamlit run main.py`
2. The system will automatically look for PDFs in the `./pdfs` folder
3. If no saved index exists, the system will process all PDF files and create embeddings
4. Once processing is complete, you can start querying your documents

### Querying Documents

1. Use the chat interface to ask questions about your PDF documents
2. The system will search through all processed documents to find relevant information
3. Responses include:
   - AI-generated answers based on document content
   - Source document citations
   - Confidence scores for retrieval quality

### Advanced Features

- **Document Statistics**: View processing metrics in the sidebar
- **Advanced Search**: Use the expandable search section for custom queries
- **Chat History**: Previous conversations are maintained during the session
- **Confidence Scoring**: Each response includes a confidence indicator

### Example Queries

- "What are the key compliance requirements mentioned in the documents?"
- "Explain the tax rate structures described in the PDFs"
- "What are the accounting standards discussed?"
- "Summarize the private company compliance requirements"

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Required Google Gemini API key for AI responses

### Customizable Parameters

- **Chunk Size**: Default 500 words (adjustable in PDFProcessor)
- **Chunk Overlap**: Default 50 words for context continuity
- **Top-K Retrieval**: Default 5 most relevant chunks
- **Embedding Model**: Default "all-MiniLM-L6-v2" (changeable in RAGPipeline)

## File Structure

```
ca_domain_ai_assistant/
├── main.py                 # Main application code
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .env                   # Environment variables (create this)
├── pdfs/                  # PDF documents folder
│   ├── document1.pdf
│   └── document2.pdf
└── indexes/               # Generated vector indexes
    ├── pdf_index.index
    └── pdf_index.metadata
```

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key libraries include:

- `streamlit`: Web application framework
- `sentence-transformers`: Text embedding generation
- `faiss-cpu`: Vector similarity search
- `google-generativeai`: Google Gemini API integration
- `PyPDF2`: PDF text extraction
- `python-dotenv`: Environment variable management

---

**Note**: This application is designed specifically for Chartered Accountancy domain documents but can be adapted for other document types with minimal modifications.
