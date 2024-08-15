from flask import Flask, request, jsonify, render_template
import tempfile
import openai
import os
import json
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chat.html')

def get_api_key(file_path, key):
    with open(file_path, 'r') as file:
        data = json.load(file)
        api_key = data.get(key)
        return api_key

def extract_text_from_pdfs(pdf_files):
    extracted_documents = []
    for pdf_file in pdf_files:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            pdf_file.save(temp_file.name)
            temp_file_path = temp_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()
        extracted_documents.extend(pages)
    return extracted_documents

def build_faiss_index(text_splits, embeddings):
    vectordb = FAISS.from_documents(text_splits, embeddings)
    return vectordb

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return "No files uploaded", 400
    
    global pdf_texts, embeddings, index, document_mapping
    pages = extract_text_from_pdfs(uploaded_files)
    
    # Split the text and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    text_splits = text_splitter.split_documents(pages)
    document_mapping = text_splits  # Store splits for reference
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Build Faiss index
    index = build_faiss_index(text_splits, embeddings)
    
    return "Files uploaded and processed successfully!", 200

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    user_input = request.json['message']
    
    # If the user has not uploaded documents, prompt them to do so
    if index is None:
        return jsonify({"response": "Please upload PDF files first to get started. I need documents to assist you with your queries."})
    
    # If the user hasn't asked anything yet, introduce the assistant
    if user_input.lower() in ['hi', 'hello', 'hey', 'start']:
        introduction = (
            "Hello! I am your document assistant. I can help you answer questions based on the documents you upload. "
            "Please upload your PDF files, and I'll assist you in extracting information and answering your queries."
        )
        return jsonify({"response": introduction})
    
   
    
    
    
    
    
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature = 0.0)

    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.as_retriever(search_type="similarity", k=1), # or you can pass other types for example `compression_retriever`
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)   
    question = user_input
    result = qa_chain.invoke({"query": question})
    print(result)
    
    return jsonify({"response": result["result"]})

if __name__ == '__main__':
    global pdf_texts, embeddings, index, document_mapping, OPENAI_API_KEY
    # Store uploaded PDFs and their extracted content
    pdf_texts = []
    embeddings = None
    index = None
    document_mapping = []
    OPENAI_API_KEY = get_api_key('api_key.json', 'openai_api_key')
    openai.api_key = OPENAI_API_KEY
    # Alternatively, set the API key as an environment variable
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    app.run(debug=True)
