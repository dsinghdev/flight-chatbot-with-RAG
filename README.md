# Flight Chatbot with Streamlit

A conversational chatbot built with Streamlit to assist customers with flight information and personalized queries. 

## Setup Instructions

### 1. Install Required Packages

First, create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  
```

Then, install the dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Make sure to add your Hugging Face API token and Pinecone API key as environment variables for secure access. You can do this by creating a .env file in the project root with the following format:

```bash
HUGGINGFACE_API_KEY=your_hugging_face_token_here
PINECONE_API_KEY=your_pinecone_key_here
```

Alternatively, you can set these directly in your terminal or development environment.

### 3. Run the App
Once the environment is set up, start the Streamlit app by running:

```bash
streamlit run app.py
```
This command will start the app on a local server, typically at http://localhost:8501. Open this URL in your browser to interact with the chatbot.