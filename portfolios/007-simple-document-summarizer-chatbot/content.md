# Simple Document Summarizer Chatbot using Google Gemini API
----

<em><a href="https://github.com/PranayJagtap06/simple-doc-summarizer-chatbot" target="_blank" rel="noopener noreferrer">Check out my Simple Document Summarizer Chatbot project on GitHub!</a></em>

<em><a href="https://pranayj97-gemini-chatbot-apiapp.hf.space/" target="_blank" rel="noopener noreferrer">Visit Project's Streamlit App</a></em>

<!-- ## Table of Contents -->
<!-- ---- -->
<!---->
<!--   - [1. Overview](#1-overview) -->
<!--   - [2. Features](#2-features) -->
<!--   - [3. System Architecture](#3-system-architecture) -->
<!--   - [4. Prerequisites](#4-prerequisites) -->
<!--   - [5. Installation Guide](#5-installation-guide) -->
<!--     - [5.1 System Dependencies](#5.1-system-dependencies) -->
<!--     - [5.2 Python Environment Setup](#5.2-python-environment-setup) -->
<!--     - [5.3 Environment Configuration](#5.3-environment-configuration) -->
<!--   - [6. Project Structure](#6-project-structure) -->
<!--   - [7. Configuration](#7-configuration) -->
<!--   - [8. Running the Application](#8-running-the-application) -->
<!--     - [Method 1: Direct Execution](#method-1-direct-execution) -->
<!--     - [Method 2: Docker Deployment](#method-2-docker-deployment) -->
<!--       - [A. Single Docker Container](#a-single-docker-container) -->
<!--       - [B. Using Docker Compose](#b-using-docker-compose) -->
<!--   - [9. Usage Guide](#9-usage-guide) -->
<!--     - [9.1 Document Upload](#9.1-document-upload) -->
<!--     - [9.2 Querying Documents](#9.2-querying-documents) -->
<!--     - [9.3 Document Management](#9.3-document-management) -->
<!--   - [10. API Documentation](#10-api-documentation) -->
<!--     - [10.1 Key Endpoints](#10.1-key-endpoints) -->
<!--   - [11. Testing](#11-testing) -->
<!--   - [12. Troubleshooting](#12-troubleshooting) -->
<!--     - [12.1 Common Issues](#12.1-common-issues) -->
<!--   - [13. Performance Optimization](#13-performance-optimization) -->
<!--   - [14. Security Considerations](#14-security-considerations) -->

## **1. Overview**
----
The Document Research & Summarizer Chatbot is an AI-powered application that processes multiple documents, extracts information, and identifies themes across documents. It uses advanced `NLP` techniques to understand document content and provide intelligent responses to user queries. Basically, utilizes ***`Google Gemini LLM`*** under the hood.

## **2. Features**
----
- **Document Processing**
  - Support for multiple file formats (PDF, TXT, PNG, JPG, JPEG)
  - OCR capabilities for scanned documents and images
  - Text extraction and preprocessing
  - Automatic paragraph segmentation

- **Search & Analysis**
  - Vector-based document search
  - Natural language query processing
  - Theme identification across documents
  - Citation tracking (document, page, paragraph)

- **User Interface**
  - Streamlit-based web interface
  - Document library management
  - Chat history tracking
  - Real-time response generation

## **3. System Architecture**
----
- **Frontend**: Streamlit web application
- **Backend**: FastAPI REST API
- **AI Model**: Google Gemini Pro
- **Vector Database**: ChromaDB
- **OCR Engine**: Tesseract
- **Text Processing**: Sentence Transformers

## **4. Prerequisites**
----
- Python 3.8 or higher
- Tesseract OCR
- Google AI Studio API key
- Docker (optional)
- 2GB+ RAM
- 5GB+ disk space

## **5. Installation Guide**
----

### **5.1 System Dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr libmagic1 wget neovim docker

# Arch
sudo pacman -Syyu
sudo pacman -S tesseract tesseract-data-eng tesseract-data-osd wget neovim docker

# MacOS
brew install tesseract docker
```

### **5.2 Python Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### **5.3 Environment Configuration**
Create a `.env` file in the backend directory:
```env
GEMINI_API_KEY=your_api_key_her
```

Or `export GEMINI_API_KEY=your_api_key_here` --for Linux/MacOS


## **6. Project Structure**
----

```pl
Simple-Doc-Summarizer-Chatbot/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI App/endpoints
│   │   ├── core/         # Core utilities
│   │   ├── models/       # Schemas
│   │   ├── services/     # App services
│   │   ├── main.py       # Streamlit App
│   │   └── config.py     # Configuration
│   ├── data/             # Data storage
│   ├── requirements.txt
│   └── Dockerfile
├── tests/                # Test suite
├── docs/                 # Documentation
├── demo/                 # Demo scripts
└── README.md
```

## **7. Configuration**
----
Key configuration options in `backend/app/config.py`:
```python
ALLOWED_EXTENSIONS = {"pdf", "txt", "png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## **8. Running the Application**
----

### **Method 1: Direct Execution**
 -  Start the backend server:
    ```bash
    cd backend
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300 --reload
    ```

 - Start the Streamlit frontend (in a new terminal):
    ```bash
    streamlit run main.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS true --server.enableXsrfProtection true
    ```

### **Method 2: Docker Deployment**
 - #### *A. Single Docker Container*
    ```bash
    cd backend

    # Build the Docker image
    docker build -t chatbot-theme-identifier .

    # Run the container
    docker run -p 8501:8501 \
      --env-file ./.env \
      chatbot-theme-identifier
    ```

 - #### *B. Using Docker Compose*
    ```bash
    cd backend

    # Build the service and
    docker compose up -w --menu --build
    ```

## **9. Usage Guide**
----

### **9.1 Document Upload**
 - Use the sidebar to upload documents
 - Supported formats: PDF, TXT, PNG, JPG, JPEG
 - Click "Process Documents" to initiate processing

### **9.2 Querying Documents**
 - Enter your question in the main chat interface
 - Click "Search" to process the query
 - View document-specific answers and identified themes
 - Browse chat history with expandable responses

### **9.3 Document Management**
- Use the Document Library tab to view all uploads
- Filter documents by type
- Sort by upload time, filename, or size
- Preview document content

## **10. API Documentation**
----

### **10.1 Key Endpoints**

 - Document Upload
    ```bash
    POST /api/v1/documents/upload
    Content-Type: multipart/form-data
    ```

- Chat Query
    ```bash
    POST /api/v1/chat/query
    Content-Type: application/json
    {
        "query": "your question here"
    }
    ```

- Vector Search
    ```bash
    POST /api/v1/vector/search
    Content-Type: application/json
    {
        "query": "search text",
        "n_results": 20
    }
    ```

## **11. Testing**
----

Run the test suite:
```bash
cd tests
pytest -v
```

## **12 Troubleshooting**
----

### **12.1 Common Issues**

 - **OCR Not Working**
   - Verify Tesseract installation
   - Check file permissions
   - Ensure supported image format

 - **API Connection Errors**
   - Verify API key in .env file
   - Check network connectivity
   - Confirm correct ports are exposed

 - **Memory Issues**
   - Reduce batch size
   - Limit concurrent uploads
   - Clear vector store cache

## **13. Performance Optimization**
----

 - Use batch processing for multiple documents
 - Enable caching for frequent queries
 - Optimize chunk size based on document length
 - Adjust vector search parameters

## **14. Security Considerations**
----

 - API key protection
 - Rate limiting
 - Input validation
 - Secure file handling
 - CORS configuration
