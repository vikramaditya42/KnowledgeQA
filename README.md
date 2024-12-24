# Enhanced PDF-Based Question Answering System

This project implements a PDF-based Question Answering (QA) system using the **llama3.1** model via **Ollama**. Users can upload PDFs, ask questions based on the context of the documents, and provide feedback to improve the system's accuracy over time. The system also supports metadata-based document filtering and retraining embeddings using user feedback.

---

## **Architecture**

The architecture consists of the following components:

1. **Document Processing**:
   - Uploaded PDFs are processed using `PyPDFLoader` to split them into pages.
   - Each page is embedded into a vector space using `OllamaEmbeddings`.
   - The embeddings are stored in a FAISS vector store for fast similarity-based retrieval.

2. **Question Answering Pipeline**:
   - A custom RAG (Retrieve and Generate) pipeline is used:
     - Relevant documents are retrieved using the FAISS vector store.
     - A prompt template is filled with the retrieved context and the user’s question.
     - The **llama3.1** model via **Ollama** generates an answer based on the prompt.

3. **Feedback Mechanism**:
   - Users can provide feedback, including corrections to answers.
   - Feedback is logged in a `Pandas` DataFrame for later analysis.
   - The system can retrain embeddings using the corrected answers provided as feedback.

4. **Metadata Management**:
   - Metadata (tags, categories, and timestamps) is stored for each document to enable advanced filtering during retrieval.

5. **Gradio Interface**:
   - The user interface is built with Gradio, allowing interaction through file uploads, text input, and table management.

---

## **Tools and Technologies**

1. **Python Libraries**:
   - **LangChain**:
     - `PyPDFLoader` for processing PDFs.
     - `OllamaEmbeddings` and **llama3.1** model for text embeddings and LLM-based answer generation.
     - `FAISS` for vector storage and similarity-based document retrieval.
   - **Pandas**:
     - Used for managing feedback data.
   - **Gradio**:
     - Provides an interactive web-based interface for the application.
   - **FAISS**:
     - Fast similarity search for efficient document retrieval.

2. **Ollama**:
   - **llama3.1** model for answering questions and generating embeddings.

3. **Prompt Template**:
   - The system uses a structured template for the RAG pipeline to ensure consistency and relevance in responses.

4. **Core Libraries**:
   - `datetime`, `uuid`, `os` for metadata and unique ID generation.

---

## **Features**

- **Upload PDFs**: Users can upload PDF documents for question answering.
- **Ask Questions**: Retrieve answers based on the content of the uploaded PDFs.
- **Filter by Metadata**: Use tags and categories to filter documents during retrieval.
- **Feedback Loop**: Users can provide feedback, which is used to retrain the embeddings.
- **Interactive UI**: Gradio provides an easy-to-use web interface.
- **Retrain on Feedback**: Allows retraining embeddings using corrected answers provided by users.

---

