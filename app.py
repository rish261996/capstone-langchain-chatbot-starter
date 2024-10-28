from flask import Flask, render_template
from flask import request, jsonify, abort
import logging

from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
import os

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Set Flask's log level to DEBUG
app.logger.setLevel(logging.DEBUG)

global chatbot_llm_chain
global knowledgebase_llm
global knowledgebase_qa

# Setup chatbot LLM
def setup_chatbot_llm():
    global chatbot_llm_chain
    template = """
    You are a chatbot that had a conversation with a human. Consider the previous conversation to answer the new question.

    Previous conversation:{chat_history}
    New human question: {question}

    Response:"""
    
    prompt = PromptTemplate(template=template, input_variables=["question", "chat_history"])
    llm = Cohere(cohere_api_key=os.environ.get("COHERE_API_KEY"))
    memory = ConversationBufferMemory(memory_key="chat_history")
    chatbot_llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, memory=memory)
    app.logger.debug('Chatbot LLM chain set up successfully')

# Setup knowledge base LLM
def setup_knowledgebase_llm():
    global knowledgebase_qa
    app.logger.debug('Setting up knowledge base')
    try:
        # Load embeddings and vectorstore
        embeddings = CohereEmbeddings(cohere_api_key=os.environ.get("COHERE_API_KEY"))
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        
        knowledgebase_qa = RetrievalQA.from_chain_type(
            llm=Cohere(cohere_api_key=os.environ.get("COHERE_API_KEY")),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        app.logger.debug('Knowledge base setup successful')
    except Exception as e:
        app.logger.error(f"Error setting up knowledge base: {str(e)}")

# General setup function
def setup():
    setup_chatbot_llm()
    setup_knowledgebase_llm()

# Answer query from knowledge base
def answer_from_knowledgebase(message):
    global knowledgebase_qa
    if not knowledgebase_qa:
        app.logger.error('Knowledge base is not set up')
        abort(500, description="Knowledge base not available")
    try:
        res = knowledgebase_qa({"query": message})
        app.logger.debug('Query successful')
        return res['result']
    except Exception as e:
        app.logger.error(f"Error during KB query: {str(e)}")
        abort(500, description="Query failed")

# Search knowledge base and retrieve sources
def search_knowledgebase(message):
    global knowledgebase_qa
    if not knowledgebase_qa:
        app.logger.error('Knowledge base is not set up')
        abort(500, description="Knowledge base not available")
    try:
        res = knowledgebase_qa({"query": message})
        if 'source_documents' not in res:
            app.logger.error('No source documents found')
            return "No sources available"
        
        sources = ""
        for count, source in enumerate(res['source_documents'], 1):
            sources += f"Source {count}\n{source.page_content}\n"
        return sources
    except Exception as e:
        app.logger.error(f"Error during KB search: {str(e)}")
        abort(500, description="Search failed")

# Chatbot LLM function
def answer_as_chatbot(message):
    template = """Question: {question}
    Answer as if you are an expert Python developer"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = Cohere(cohere_api_key=os.environ.get("COHERE_API_KEY"))
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    try:
        res = llm_chain.run(message)
        app.logger.debug('Chatbot query successful')
        return res
    except Exception as e:
        app.logger.error(f"Error during chatbot query: {str(e)}")
        abort(500, description="Chatbot query failed")

# API routes
@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json.get('message')
    
    if not message:
        abort(400, description="Message is required")
    
    # Generate a response
    response_message = answer_from_knowledgebase(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

@app.route('/search', methods=['POST'])
def search():    
    message = request.json.get('message')
    
    if not message:
        abort(400, description="Message is required")
    
    # Generate a response
    response_message = search_knowledgebase(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json.get('message')
    
    if not message:
        abort(400, description="Message is required")
    
    # Generate a response
    response_message = answer_as_chatbot(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

# Home route
@app.route("/")
def index():
    return render_template("index.html", title="")

# Initialize the app on startup
if __name__ == "__main__":
    app.logger.debug('Initializing LLM chains...')
    setup()
    app.logger.debug('LLM chains initialized')
    app.run(host='0.0.0.0', port=5001)
