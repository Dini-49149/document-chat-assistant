from flask import Flask, request, jsonify, render_template
import tempfile
import openai
import os
import json
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

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
        return jsonify({"response": "No files uploaded"}), 400
    
    global pdf_texts, embeddings, index, document_mapping

    # Extract chunk size, chunk overlap, length function, and text splitter type from form data
    chunk_size = int(request.form.get("chunk_size", 1000))
    chunk_overlap = int(request.form.get("chunk_overlap", 200))
    length_function = request.form.get("length_function", "len")
    text_splitter_type = request.form.get("text_splitter_type", "character")

    # Set the length function
    if length_function == "len":
        length_func = len
    elif length_function == "token":
        # Define your token length function here
        def token_length(text):
            return len(text.split())  # Example: count the number of words as tokens
        
        length_func = token_length
    else:
        return jsonify({"response": "Invalid length function specified"}), 400

    pages = extract_text_from_pdfs(uploaded_files)
    
    # Determine text splitter based on user selection
    if text_splitter_type == "character":
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_func
        )
    elif text_splitter_type == "recursive_character":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_func
        )
    text_splits = text_splitter.split_documents(pages)
    document_mapping = text_splits  # Store splits for reference
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    global index
    # Build Faiss index
    index = build_faiss_index(text_splits, embeddings)
    
    # Return a JSON response that can be displayed in the chat
    return jsonify({"response": "Files uploaded and processed successfully!"}), 200




def route_and_response(user_input):
    # Initialize your language model (replace with appropriate initialization)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # Define the paraphrase chain
    paraphrase_chain = LLMChain(
        prompt=PromptTemplate.from_template(
            """You are a helpful assistant in Paraphrasing the given instruction.

    Instruction: {instruction}
    Answer:"""
        ),
        llm=llm,
        output_parser=StrOutputParser(),
    )
    
    # Define the General Chain (Non-document questions)
    general_chain = LLMChain(
        prompt=PromptTemplate.from_template(
            """You are a helpful assistant in answering questions of users related to the document uploaded. Answer the following question:

    Question: {question}
    Answer:"""
        ),
        llm=llm,
        output_parser=StrOutputParser(),
    )

    # Routing Chain to classify input as question or not
    routing_chain = LLMChain(
        prompt=PromptTemplate.from_template(
            """The user may ask questions related to the content of documents they have uploaded, or they may ask general questions unrelated to the documents.
            Given the user input below, classify it as either being `document-related` or `general`.

    Do not respond with more than one word.

    <user_input>
    {user_input}
    </user_input>

    Classification:"""
        ),
        llm=llm,
        output_parser=StrOutputParser(),
    )
    # Classify the input
    classification = routing_chain.invoke({"user_input": user_input})

    # Determine which chain to route the input to
    if classification['text'].lower() == "document-related":
        if index:
            # Build prompt
            template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=index.as_retriever(search_type="similarity", k=3),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
            
            question = user_input
            result = qa_chain.invoke({"query": question})
            
            return result["result"]
        else:
            question = "Please upload PDF files first to get started. I need documents to assist you with your queries."
            paraphrase_result = paraphrase_chain.invoke({"instruction": question})
            return paraphrase_result['text']
    
    else:
        general_response = general_chain.invoke({"question": user_input})
        return general_response['text']


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    user_input = request.json['message']
    
    result = route_and_response(user_input)
    return jsonify({"response": result})

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
