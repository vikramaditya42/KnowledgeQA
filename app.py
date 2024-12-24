import os
import gradio as gr
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from uuid import uuid4
import pandas as pd

# Initialize model and embeddings
MODEL = "llama3.1:8b"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Metadata-enabled vector store
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
metadata_store = {}  # Store metadata for documents (keyed by UUID)

# Feedback store as a Pandas DataFrame
feedback_store = pd.DataFrame(columns=["Question", "Correct Answer", "Feedback"])

# Prompt template
rag_template = """
Answer the question based only on the following context. 
If you can't answer the question based on the context, reply 
"Sorry. I don't know at this moment":

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(rag_template)
parser = StrOutputParser()

# Functions
def upload_and_process_pdf(file, tags, category):
    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()
    
    uuids = [str(uuid4()) for _ in range(len(pages))]
    vector_store.add_documents(documents=pages, ids=uuids)
    
    # Add metadata
    timestamp = datetime.now().isoformat()
    for uuid in uuids:
        metadata_store[uuid] = {"tags": tags, "category": category, "timestamp": timestamp}
    
    return f"PDF processed successfully with {len(pages)} pages!"

def ask_question(question, filter_tags=None, filter_category=None):
    retriever = vector_store.as_retriever()
    filtered_contexts = []
    
    # Apply metadata filters
    for uuid, metadata in metadata_store.items():
        if filter_tags and not any(tag in metadata["tags"] for tag in filter_tags.split(",")):
            continue
        if filter_category and metadata["category"] != filter_category:
            continue
        
        # Invoke retriever and ensure it appends text
        retrieved = retriever.invoke(question)
        if isinstance(retrieved, list):
            # Extract content from Document objects
            filtered_contexts.extend([doc.page_content for doc in retrieved if hasattr(doc, "page_content")])
        elif isinstance(retrieved, str):
            filtered_contexts.append(retrieved)
    
    if not filtered_contexts:
        return "No matching context found for the specified filters."
    
    context = "\n\n".join(filtered_contexts)
    chain = (
        {
            "context": lambda x: context,
            "question": lambda x: question,
        }
        | prompt
        | model
        | parser
    )
    return chain.invoke({"question": question})


def submit_feedback(question, correct_answer, user_feedback):
    global feedback_store
    new_feedback = pd.DataFrame(
        [{"Question": question, "Correct Answer": correct_answer, "Feedback": user_feedback}]
    )
    feedback_store = pd.concat([feedback_store, new_feedback], ignore_index=True)
    return "Feedback submitted successfully!"


def view_feedback():
    global feedback_store
    return feedback_store

def delete_feedback(index):
    global feedback_store
    if index >= 0 and index < len(feedback_store):
        feedback_store = feedback_store.drop(index).reset_index(drop=True)
        return "Feedback entry deleted successfully!"
    else:
        return "Invalid index. Please try again."

from langchain_core.documents import Document

def retrain_embeddings():
    global feedback_store, vector_store
    
    # Prepare documents for retraining
    retrained_data = feedback_store[["Correct Answer"]].dropna()
    retrained_uuids = [str(uuid4()) for _ in range(len(retrained_data))]
    retrained_docs = [
        Document(page_content=ans, metadata={"source": "feedback"})
        for ans in retrained_data["Correct Answer"]
    ]
    
    # Add documents to vector store
    vector_store.add_documents(documents=retrained_docs, ids=retrained_uuids)
    return "Retraining completed successfully!"


# Gradio app
with gr.Blocks() as qa_demo:
    gr.Markdown("### Enhanced PDF-Based Question Answering System with Knowledge Management and Feedback")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            tags_input = gr.Textbox(label="Tags (comma-separated)")
            category_input = gr.Textbox(label="Category")
            process_button = gr.Button("Process PDF")
            status_output = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column():
            question_input = gr.Textbox(label="Ask a Question")
            filter_tags_input = gr.Textbox(label="Filter by Tags (optional)")
            filter_category_input = gr.Textbox(label="Filter by Category (optional)")
            answer_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer", interactive=False)
    
    with gr.Row():
        gr.Markdown("### Feedback Section")
        feedback_question_input = gr.Textbox(label="Question")
        correct_answer_input = gr.Textbox(label="Correct Answer")
        user_feedback_input = gr.Textbox(label="Your Feedback")
        feedback_button = gr.Button("Submit Feedback")
        feedback_status_output = gr.Textbox(label="Feedback Status", interactive=False)
    
    with gr.Row():
        gr.Markdown("### Feedback Management")
        feedback_table = gr.Dataframe(label="Feedback Entries")
        update_feedback_button = gr.Button("Reload Feedback Table")
        feedback_index_input = gr.Number(label="Delete Feedback at Index", precision=0)
        delete_feedback_button = gr.Button("Delete Feedback")
        delete_feedback_status_output = gr.Textbox(label="Delete Status", interactive=False)
        retrain_button = gr.Button("Retrain Embeddings")
        retrain_status_output = gr.Textbox(label="Retrain Status", interactive=False)
    
    # Button actions
    process_button.click(
        upload_and_process_pdf,
        inputs=[pdf_input, tags_input, category_input],
        outputs=status_output,
    )
    answer_button.click(
        ask_question,
        inputs=[question_input, filter_tags_input, filter_category_input],
        outputs=answer_output,
    )
    feedback_button.click(
        submit_feedback,
        inputs=[feedback_question_input, correct_answer_input, user_feedback_input],
        outputs=feedback_status_output,
    )
    update_feedback_button.click(view_feedback, outputs=feedback_table)
    delete_feedback_button.click(delete_feedback, inputs=feedback_index_input, outputs=delete_feedback_status_output)
    retrain_button.click(retrain_embeddings, outputs=retrain_status_output)

qa_demo.launch()
