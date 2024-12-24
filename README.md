Enhanced PDF-Based Question Answering System
This project implements a PDF-based Question Answering (QA) system using advanced language models and embedding techniques. Users can upload PDFs, ask questions based on the context of the documents, and provide feedback to improve the system's accuracy over time. The system also supports metadata-based document filtering and retraining embeddings using user feedback.

Architecture
The architecture consists of the following components:

Document Processing:

Uploaded PDFs are processed using PyPDFLoader to split them into pages.
Each page is embedded into a vector space using OllamaEmbeddings.
The embeddings are stored in a FAISS vector store for fast similarity-based retrieval.
Question Answering Pipeline:

A custom RAG (Retrieve and Generate) pipeline is used:
Relevant documents are retrieved using the FAISS vector store.
A prompt template is filled with the retrieved context and the user’s question.
The Ollama language model generates an answer based on the prompt.
Feedback Mechanism:

Users can provide feedback, including corrections to answers.
Feedback is logged in a Pandas DataFrame for later analysis.
The system can retrain embeddings using the corrected answers provided as feedback.
Metadata Management:

Metadata (tags, categories, and timestamps) is stored for each document to enable advanced filtering during retrieval.
Gradio Interface:

The user interface is built with Gradio, allowing interaction through file uploads, text input, and table management.
Tools and Technologies
Python Libraries:

LangChain:
PyPDFLoader for processing PDFs.
OllamaEmbeddings and Ollama model for text embeddings and LLM-based answer generation.
FAISS for vector storage and similarity-based document retrieval.
Pandas:
Used for managing feedback data.
Gradio:
Provides an interactive web-based interface for the application.
FAISS:
Fast similarity search for efficient document retrieval.
Prompt Template:

The system uses a structured template for the RAG pipeline to ensure consistency and relevance in responses.
Core Libraries:

datetime, uuid, os for metadata and unique ID generation.
Features
Upload PDFs: Users can upload PDF documents for question answering.
Ask Questions: Retrieve answers based on the content of the uploaded PDFs.
Filter by Metadata: Use tags and categories to filter documents during retrieval.
Feedback Loop: Users can provide feedback, which is used to retrain the embeddings.
Interactive UI: Gradio provides an easy-to-use web interface.
Retrain on Feedback: Allows retraining embeddings using corrected answers provided by users.
Deployment Steps
Follow these steps to deploy the system:

1. Clone the Repository
bash
Copy code
git clone <repository-url>
cd <repository-directory>
2. Set Up the Python Environment
Ensure you have Python 3.8+ installed. Create a virtual environment and install the dependencies.

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Run the Application
Launch the Gradio app.

bash
Copy code
python app.py
The application will start locally and provide a link to access the Gradio interface.

Feedback and Logs
Feedback:

Users can submit feedback directly in the Gradio app.
Feedback is stored in a Pandas DataFrame for further analysis.
Logging:

User queries and system responses are logged to a query_logs.txt file for debugging and performance evaluation.
Future Enhancements
Add Export Options:
Enable exporting feedback and metadata to CSV for external analysis.
Performance Metrics:
Track accuracy and response times directly in the interface.
Scalability:
Deploy the app using cloud services (e.g., AWS, GCP) for higher availability.
