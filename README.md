# Document Chat Assistant

**Document Chat Assistant** is a Flask-based application that allows users to upload PDF documents and interact with a chatbot that can answer questions based on the content of the uploaded files. This project leverages advanced document processing, embedding generation, and a retrieval-based question-answering system using LangChain and OpenAI's GPT models.

## Features

- **PDF Upload and Processing**: Users can upload PDF files, which are processed to extract and split the content for efficient querying.
- **Interactive Chat Interface**: The chatbot interacts with the user, answering questions based on the content of the uploaded PDF files.
- **Responsive Feedback**: Once files are uploaded, the system provides immediate feedback, and the chatbot is ready to answer queries.

## Folder Structure

```
document-chat-assistant/
│
├── app.py                   # Main application file containing Flask routes and backend logic
├── templates/
│   └── chat.html            # Frontend interface for chat and file upload
├── api_key.json             # OpenAI API key configuration (not in version control)
└── requirements.txt         # List of Python dependencies
```

### Detailed Explanation of Components

- **app.py**: This is the main Python file that serves as the backend for the application. It handles routing, file uploads, and the interaction with the Language Model, LangChain and FAISS libraries.
- **templates/chat.html**: This HTML file defines the frontend of the application. It includes the chat interface and the file upload feature, allowing users to upload PDFs and interact with the chatbot.
- **api_key.json**: This file contains your OpenAI API key, which is essential for generating embeddings and responses. (Ensure this file is not included in version control for security reasons.)
- **requirements.txt**: This file lists all the Python dependencies required to run the application. 

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher installed on your local machine.
- Flask and required Python libraries installed (listed in `requirements.txt`).
- An OpenAI API key stored in `api_key.json`.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/document-chat-assistant.git
   cd document-chat-assistant

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add your OpenAI API key:
Create a file named `api_key.json` in the root of your project directory and add your OpenAI API key:

   ```json
   {
       "openai_api_key": "your-openai-api-key"
   }
   ```

### Running the Application

1. **Start the Flask application:**

   ```bash
   python app.py
   ```

2. **Access the application:**
Open your web browser and navigate to `http://127.0.0.1:5000/` to interact with the Document Chat Assistant.

### Project Flow
1. **Upload Files**: Users upload one or more PDF files using the web interface.
2. **Processing**: The uploaded files are processed using LangChain to split the text into chunks. These chunks are then embedded using OpenAI's embeddings.
3. **Querying**: Users can ask questions in the chat interface. The system retrieves the most relevant chunk from the vector database and generates a response using GPT-3.5-turbo.

### Future Enhancements

- **Enhanced Document Support**: Expand the document types supported beyond PDFs (e.g., Word, Excel).
- **Backend Improvements**: Introduce additional backend features such as document management, caching, and performance monitoring.
- **Deployment**: Package the application with Docker for easier deployment to cloud platforms like AWS, Azure, or Heroku.

### Contributing

We welcome contributions from the community. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a Pull Request.

### License

This project is licensed under the MIT License. See the `LICENSE
