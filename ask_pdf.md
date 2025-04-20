Okay, let's create a tutorial document based on the code and concepts we've developed.

---

# Tutorial: Building a PDF Analyzer App with LLMs, LangChain, and RAG

**(Generated on: Tuesday, April 15, 2025)**

## Introduction

Welcome! This tutorial will guide you through building a web application that can understand and interact with the content of PDF documents. We'll create an app that can:

1.  **Summarize** the key information within a PDF.
2.  Answer **questions** about the PDF's content (Question & Answering).
3.  Generate a **Knowledge Graph** visualizing the main entities and relationships mentioned.

To achieve this, we'll leverage the power of Large Language Models (LLMs) via the OpenAI API, orchestrate the workflow using the **LangChain** framework, implement the **Retrieval-Augmented Generation (RAG)** pattern for accurate, context-aware responses, and build an interactive user interface with **Gradio**.

**What you will learn:**

* Fundamental concepts of LLMs, LangChain, and RAG.
* How to load and process text data from PDF files.
* How to generate text embeddings and use vector stores for retrieval.
* Implementing RAG for context-aware Q&A.
* Using LLMs for summarization and structured data extraction (Knowledge Graphs).
* Building a simple web UI for your LLM application using Gradio.

**Why this approach?**

LLMs are powerful but often lack knowledge of specific, private, or very recent documents. RAG allows us to bridge this gap by providing the LLM with relevant information retrieved directly from *your* data (in this case, a PDF) before it generates a response. LangChain simplifies the complex process of connecting these different components (data loading, embedding, retrieval, LLM interaction).

## Prerequisites

Before we start coding, ensure you have the following:

1.  **Python:** Version 3.8 or higher installed.
2.  **OpenAI API Key:** You'll need an account with OpenAI and an API key. You can get one from [platform.openai.com](https://platform.openai.com/).
3.  **Required Libraries:** Install the necessary Python packages using pip:
    ```bash
    pip install langchain langchain-openai openai faiss-cpu pypdf python-dotenv networkx pyvis matplotlib tiktoken gradio
    # Use faiss-gpu instead of faiss-cpu if you have a compatible NVIDIA GPU and CUDA installed
    ```
4.  **Environment Setup:** Create a file named `.env` in your project directory and add your OpenAI API key like this:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *Never commit your `.env` file or API keys directly into version control (like Git).*
5.  **A Sample PDF:** Have a PDF file ready for testing (e.g., `mydocument.pdf`).

## Core Concepts Explained

Let's briefly understand the key technologies:

1.  **Large Language Models (LLMs):** These are sophisticated AI models (like OpenAI's GPT-3.5 or GPT-4) trained on vast amounts of text data. They excel at understanding and generating human-like text for tasks like summarization, translation, and answering questions. We'll use the OpenAI API to access these models.
2.  **LangChain:** An open-source framework designed to simplify the development of applications powered by LLMs. It provides modular components and "chains" to combine these components into workflows. Key components we'll use include:
    * **Document Loaders:** To load data from sources (like PDFs). (`PyPDFLoader`)
    * **Text Splitters:** To break large documents into smaller chunks. (`RecursiveCharacterTextSplitter`)
    * **Embeddings:** To convert text chunks into numerical vectors capturing semantic meaning. (`OpenAIEmbeddings`)
    * **Vector Stores:** Databases optimized for storing and searching vectors based on similarity. (`FAISS`)
    * **Retrievers:** To fetch relevant chunks from the vector store based on a query.
    * **LLMs/Chat Models:** Wrappers to interact consistently with different LLM providers. (`ChatOpenAI`)
    * **Chains:** Pre-built or custom sequences of operations (e.g., summarizing documents (`load_summarize_chain`), performing Q&A (`RetrievalQA`)).
3.  **Retrieval-Augmented Generation (RAG):** This is the core pattern for making LLMs answer questions based on specific documents.
    * **Indexing:** The PDF is loaded, split into chunks, and each chunk is embedded into a vector. These vectors are stored in a vector store (FAISS in our case).
    * **Retrieval:** When you ask a question, it's also embedded. The vector store finds the text chunks whose embeddings are most similar (semantically relevant) to the question's embedding.
    * **Generation:** The original question *and* the retrieved text chunks (the "context") are sent to the LLM. The LLM is instructed to answer the question based *only* on the provided context. This grounds the LLM's response in the document's facts, reducing hallucination and allowing it to answer questions about information not in its original training data.

    ```
    +-----------------+      +-----------------+      +-----------------+
    | PDF Document    | ---> | Load & Split    | ---> | Embed Chunks    |
    +-----------------+      +-----------------+      +-----------------+
          |                                                |
          |                                                V
    +-----------------+      +-----------------+      +-----------------+
    | User Question   | ---> | Embed Question  | ---> | Retrieve Chunks | ----> Vector Store (FAISS)
    +-----------------+      +-----------------+      +-----------------+      +-----------------+
                                                            |
                                                            V (Relevant Chunks + Question)
                                                      +-----------------+
                                                      | LLM (OpenAI)    | ---> Answer
                                                      +-----------------+
    ```
    *Simplified RAG Flow Diagram*

4.  **Gradio:** A Python library that makes it incredibly easy to create simple web UIs for machine learning models and other Python scripts. Ideal for creating demos like ours.

## Building the Backend Logic (Python Script)

Let's break down the Python code (`pdf_analyzer_app.py` from our previous steps).

**Step 1: Setup and Configuration**

```python
import os
import re
import networkx as nx
# ... other necessary imports ...
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# ... other LangChain imports ...
import gradio as gr

# --- Configuration ---
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
# ... other config variables ...

# Load API Key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")

# --- Initialize Models ---
try:
    llm = ChatOpenAI(temperature=0, model_name=LLM_MODEL) # Temperature=0 for more deterministic output
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
except Exception as e:
    print(f"Error initializing OpenAI models: {e}")
    llm = None
    embeddings = None
```

* We import all necessary libraries.
* We define configuration variables (model names, chunking parameters).
* `load_dotenv()` reads the `OPENAI_API_KEY` from the `.env` file.
* We initialize the LangChain wrappers for the OpenAI Chat Model (`ChatOpenAI`) and the Embedding model (`OpenAIEmbeddings`). Error handling is included in case the API key is missing or invalid.

**Step 2: Loading and Processing the PDF**

This happens inside the `setup_vector_store_and_docs` function, adapted for Gradio:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_vector_store_and_docs(pdf_path, progress=gr.Progress()):
    # ... (error checks for models and file path) ...
    try:
        progress(0.1, desc="Loading PDF...")
        loader = PyPDFLoader(pdf_path) # Use PyPDFLoader for PDF files
        docs_raw = loader.load() # Loads pages as 'Document' objects

        progress(0.3, desc=f"Splitting {len(docs_raw)} pages...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, # Max characters per chunk
            chunk_overlap=CHUNK_OVERLAP # Overlap helps maintain context between chunks
        )
        docs_split = text_splitter.split_documents(docs_raw)
        # ... (rest of the function) ...
```

* `PyPDFLoader` reads the PDF and creates a list of LangChain `Document` objects, usually one per page.
* LLMs have context limits, and embedding works best on smaller text pieces. Therefore, we use `RecursiveCharacterTextSplitter`. It tries to split text recursively based on characters like `\n\n`, `\n`, ` `, etc., aiming to keep related content together. `chunk_size` defines the target size, and `chunk_overlap` maintains some context between adjacent chunks.

**Step 3: Embedding and Vector Storage (Indexing for RAG)**

Continuing in `setup_vector_store_and_docs`:

```python
from langchain_community.vectorstores import FAISS

def setup_vector_store_and_docs(pdf_path, progress=gr.Progress()):
    # ... (loading and splitting) ...
    try:
        # ...
        progress(0.6, desc="Creating embeddings and vector store (FAISS)...")
        # This is the core indexing step
        vectorstore = FAISS.from_documents(docs_split, embeddings)
        progress(1.0, desc="Processing Complete!")
        print("Vector store created successfully.")
        # ... (error handling and return) ...
        return vectorstore, docs_split, "PDF processed successfully..."
```

* `FAISS.from_documents(docs_split, embeddings)` is a key LangChain operation. It takes our split document chunks, uses the `OpenAIEmbeddings` model to convert each chunk's text into a numerical vector, and stores these vectors (along with the original text) in a FAISS vector store. FAISS allows for very efficient similarity searching (finding vectors closest to a query vector).
* This function returns the created `vectorstore` and the `docs_split` for later use.

**Step 4: Implementing Summarization**

```python
from langchain.chains import load_summarize_chain

def summarize_pdf_gradio(docs, progress=gr.Progress()):
    # ... (error checks) ...
    progress(0, desc="Starting Summarization...")
    try:
        # Select the chain type based on config (e.g., map_reduce)
        summary_chain = load_summarize_chain(llm, chain_type=CHAIN_TYPE_SUMMARY)
        # Run the chain
        result = summary_chain.invoke({"input_documents": docs}) # Pass docs correctly
        summary = result.get('output_text', "Summarization failed.")
        progress(1.0, desc="Summarization Complete!")
        return summary
    # ... (error handling) ...
```

* LangChain provides `load_summarize_chain` for easy summarization.
* We use the `map_reduce` chain type here, which is suitable for large documents:
    * **Map:** It runs an initial prompt (e.g., "summarize this") on each individual text chunk.
    * **Reduce:** It takes the summaries from the "map" step and recursively combines them using another prompt until a single final summary is produced.
* The chain is invoked with the split documents, and the result contains the final summary text.

**Step 5: Implementing Q&A (RAG in action)**

```python
from langchain.chains import RetrievalQA

def query_pdf_gradio(vectorstore, query, progress=gr.Progress()):
    # ... (error checks) ...
    progress(0, desc="Performing Q&A...")
    try:
        # 1. Create a retriever from the vector store
        retriever = vectorstore.as_retriever()

        # 2. Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # 'stuff' puts all retrieved docs directly into the prompt
            retriever=retriever,
            return_source_documents=False # Optionally return which chunks were used
        )

        # 3. Run the chain with the user's query
        result = qa_chain.invoke({"query": query})
        progress(1.0, desc="Q&A Complete!")
        return result.get('result', "Failed to get an answer.")
    # ... (error handling) ...
```

* This function demonstrates the RAG pattern clearly.
* `vectorstore.as_retriever()` creates an object capable of fetching relevant documents.
* `RetrievalQA.from_chain_type` sets up the RAG workflow:
    * It takes the user's `query`.
    * Uses the `retriever` to fetch relevant document chunks from FAISS.
    * The `chain_type="stuff"` method simply "stuffs" the retrieved chunks and the question into a single prompt for the `llm`. (Other types like `map_reduce` exist for Q&A too, but `stuff` is common).
    * The LLM generates the answer based on the provided context (the retrieved chunks).
* The function returns the LLM's generated answer.

**Step 6: Implementing Knowledge Graph Extraction**

This is more experimental and involves prompting the LLM to output structured data.

```python
from langchain.prompts import PromptTemplate
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt # For static graph image

# Define the prompt for the LLM
KG_PROMPT_TEMPLATE = """
Extract knowledge triplets (Subject, Relationship, Object) from the following text...
Format each triplet on a new line, separated by '|'. Only output the triplets.
Text:
{text_chunk}
Extracted Triplets:
"""
KG_PROMPT = PromptTemplate(template=KG_PROMPT_TEMPLATE, input_variables=["text_chunk"])

# Function to call LLM for extraction per chunk
def extract_triplets(text_chunk):
    # ... (error checks) ...
    try:
        extraction_chain = KG_PROMPT | llm # Simple chain: prompt -> llm
        response = extraction_chain.invoke({"text_chunk": text_chunk})
        content = response.content # Get text output
        # Parse the LLM output (this might need refinement)
        triplets = []
        for line in content.split('\n'):
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3 and all(parts):
                    triplets.append(tuple(parts))
        return triplets
    # ... (error handling) ...

# Main function to orchestrate KG generation
def draw_knowledge_graph_gradio(docs_split, progress=gr.Progress()):
    # ... (error checks) ...
    progress(0, desc="Starting Knowledge Graph Generation...")
    all_triplets = []
    # Iterate through chunks and extract triplets
    for i, doc in enumerate(docs_split):
         # ... (update progress) ...
        chunk_text = doc.page_content
        triplets = extract_triplets(chunk_text)
        if triplets: all_triplets.extend(triplets)

    if not all_triplets: return None, None, "No triplets found."

    unique_triplets = list(set(all_triplets))

    # Build graph using NetworkX
    graph = nx.DiGraph()
    for subj, rel, obj in unique_triplets:
        graph.add_node(subj, label=subj, title=subj) # Add nodes
        graph.add_node(obj, label=obj, title=obj)
        graph.add_edge(subj, obj, label=rel, title=rel) # Add edges (relationships)

    # Generate unique filenames for outputs
    # ... (filename generation logic) ...

    # Visualize using PyVis (Interactive HTML)
    try:
        net = Network(notebook=False, height='750px', width='100%', directed=True)
        net.from_nx(graph)
        # ... (PyVis options) ...
        net.save_graph(kg_output_html)
    except Exception as e: kg_output_html = None

    # Visualize using Matplotlib (Static PNG)
    try:
        plt.figure(figsize=(16, 16))
        pos = nx.spring_layout(graph) # Calculate node positions
        nx.draw(...) # Draw nodes
        nx.draw_networkx_edge_labels(...) # Draw edge labels
        plt.savefig(kg_output_png)
        plt.close()
    except Exception as e: kg_output_png = None

    progress(1.0, desc="Knowledge Graph Generation Complete!")
    # Return file paths and status message
    return kg_output_png, kg_output_html, "Graph generated."
```

* We define a `PromptTemplate` specifically asking the LLM to extract "Subject | Relationship | Object" triplets.
* The `extract_triplets` function sends each document chunk to the LLM with this prompt.
* **Crucially:** We need to parse the LLM's text output to get the structured triplets. This parsing is basic and might need improvement depending on how consistently the LLM follows the format.
* The main `draw_knowledge_graph_gradio` function iterates through all chunks, collects triplets, uses the `networkx` library to build a graph data structure, and then uses `pyvis` (for interactive HTML) and `matplotlib` (for static PNG) to visualize the graph.
* It saves the visualizations to files and returns the file paths.

## Building the Frontend Interface (Gradio)

Now, let's wrap our backend functions in a Gradio UI.

**Step 7: Setting up the Gradio App**

```python
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft(), title="PDF Analyzer") as app:
    gr.Markdown("# PDF Analyzer: Summarize, Q&A, and Knowledge Graph")
    # ... Introduction Markdown ...

    # --- State Variables ---
    # These hold data between button clicks for the currently processed PDF
    vector_store_state = gr.State(None)
    docs_split_state = gr.State(None)

    # --- UI Layout ---
    with gr.Row(): # Arrange elements horizontally
        with gr.Column(scale=1): # Define columns for structure
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_button = gr.Button("1. Process PDF", variant="primary")
            status_textbox = gr.Textbox(label="Processing Status", interactive=False)
        with gr.Column(scale=2):
             gr.Markdown("**Instructions:** ...") # Display instructions

    # --- Tabs for Different Functions ---
    with gr.Tabs():
        with gr.TabItem("📄 Summarization"):
            summary_button = gr.Button("Get Summary", variant="secondary")
            summary_output = gr.Textbox(label="Summary", lines=15, interactive=False)

        with gr.TabItem("❓ Question Answering (Q&A)"):
            qa_question = gr.Textbox(label="Enter your question")
            qa_button = gr.Button("Ask Question", variant="secondary")
            qa_output = gr.Textbox(label="Answer", lines=10, interactive=False)

        with gr.TabItem("🕸️ Knowledge Graph"):
             kg_button = gr.Button("Generate Knowledge Graph", variant="secondary")
             kg_status_output = gr.Textbox(label="KG Status", interactive=False)
             with gr.Row():
                # Output components for KG results
                kg_image_output = gr.Image(label="Knowledge Graph (Static View)", type="filepath")
                kg_html_output = gr.File(label="Download Interactive Knowledge Graph (HTML)")
```

* We use `gr.Blocks()` for a customizable layout.
* `gr.State` is crucial. It holds Python objects (like our `vectorstore` and `docs_split`) in the app's memory between interactions, so we don't have to re-process the PDF every time we ask a question or request a summary.
* We define input components (`gr.File`, `gr.Textbox`, `gr.Button`) and output components (`gr.Textbox`, `gr.Image`, `gr.File`).
* Layout elements like `gr.Row`, `gr.Column`, and `gr.Tabs` organize the interface.

**Step 8: Connecting Backend to Frontend**

This uses the `.click()` method of Gradio buttons.

```python
    # --- Button Click Actions ---

    # When 'Process PDF' is clicked:
    process_button.click(
        fn=setup_vector_store_and_docs, # Call this backend function
        inputs=[pdf_upload],             # Pass the uploaded file object
        outputs=[vector_store_state, docs_split_state, status_textbox], # Update state and status text
        show_progress="full"             # Show Gradio's progress indicator
    )

    # When 'Get Summary' is clicked:
    summary_button.click(
        fn=summarize_pdf_gradio,
        inputs=[docs_split_state],     # Use the document chunks stored in state
        outputs=[summary_output],      # Display result in the summary textbox
        show_progress="full"
    )

    # When 'Ask Question' is clicked:
    qa_button.click(
        fn=query_pdf_gradio,
        inputs=[vector_store_state, qa_question], # Use vector store from state and the question input
        outputs=[qa_output],                      # Display answer
         show_progress="full"
    )

    # When 'Generate Knowledge Graph' is clicked:
    kg_button.click(
        fn=draw_knowledge_graph_gradio,
        inputs=[docs_split_state],                 # Use document chunks from state
        outputs=[kg_image_output, kg_html_output, kg_status_output], # Update image, file link, and status
        show_progress="full"
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    if llm and embeddings: # Only launch if models initialized correctly
        app.launch(debug=True) # Debug=True helps during development
```

* Each `.click()` call defines:
    * `fn`: The Python backend function to execute.
    * `inputs`: A list of Gradio input components whose values are passed to `fn`. `gr.State` variables are also passed as inputs.
    * `outputs`: A list of Gradio output components that will be updated with the return values of `fn`. `gr.State` variables can also be outputs if the function needs to update them.
* `show_progress="full"` tells Gradio to display its progress indicator during the function execution.
* Finally, `app.launch()` starts the web server.

## Running the Application

1.  Save the complete Python code as `pdf_analyzer_app.py`.
2.  Ensure your `.env` file is present with the API key.
3.  Open your terminal in the project directory.
4.  Run the script: `python pdf_analyzer_app.py`
5.  Open your web browser and navigate to the local URL shown in the terminal (e.g., `http://127.0.0.1:7860`).
6.  Follow the instructions in the UI: Upload a PDF, process it, then use the tabs to summarize, ask questions, or generate the knowledge graph.

## Conclusion & Next Steps

Congratulations! You've built a functional web application that uses LangChain and the RAG pattern to interact intelligently with PDF documents. You learned how to load data, create embeddings, use vector stores for retrieval, and leverage LLMs for summarization, Q&A, and even experimental knowledge extraction. You also saw how easy Gradio makes it to build an interactive demo.

**Potential Improvements & Further Learning:**

* **Error Handling:** Add more robust error handling, especially for API calls and file processing.
* **Model Selection:** Allow users to choose different LLM or embedding models via the UI.
* **Vector Store Persistence:** Use `vectorstore.save_local()` and `FAISS.load_local()` (or switch to a persistent vector database like ChromaDB or Pinecone) to avoid re-processing the same PDF every time the app restarts.
* **Advanced RAG:** Explore techniques like:
    * **Query Transformation:** Rephrasing the user's question for better retrieval.
    * **Reranking:** Using a second model to rerank the retrieved chunks for relevance before sending them to the LLM.
    * **Hybrid Search:** Combining vector search with traditional keyword search.
* **Knowledge Graph Refinement:** Improve the triplet extraction prompt, add more sophisticated parsing, or explore dedicated Named Entity Recognition (NER) and Relation Extraction models (e.g., using spaCy) for more reliable results.
* **Streaming:** Implement streaming responses for Q&A and summarization for a more responsive UI.
* **Scalability:** Optimize chunk processing for very large documents.

This project provides a solid foundation. Explore the LangChain documentation and experiment with different components and techniques to build even more sophisticated LLM-powered applications!

---