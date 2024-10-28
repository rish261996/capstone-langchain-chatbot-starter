from flask import Flask, render_template, request, jsonify
import logging
from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
import os

app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
app.logger.setLevel(logging.DEBUG)

# Global variables for LLM and knowledgebase
chatbot_llm_chain = None
knowledgebase_qa = None

# Setup function to initialize chatbot LLM and knowledgebase LLM
def setup_chatbot_llm():
    global chatbot_llm_chain
    app.logger.debug("Setting up chatbot LLM...")
    try:
        # Prompt for chatbot LLM
        template = """
        You are a chatbot that had a conversation with a human. Consider the previous conversation to answer the new question.
        
        Previous conversation: {chat_history}
        New human question: {question}
        
        Response:"""
        
        prompt = PromptTemplate(template=template, input_variables=["question", "chat_history"])
        llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
        memory = ConversationBufferMemory(memory_key="chat_history")
        chatbot_llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
        app.logger.debug("Chatbot LLM setup complete.")
    except Exception as e:
        app.logger.error(f"Error setting up chatbot LLM: {e}")

def setup_knowledgebase_llm():
    global knowledgebase_qa
    app.logger.debug("Setting up knowledgebase LLM...")
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        knowledgebase_qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        app.logger.debug("Knowledgebase LLM setup complete.")
    except Exception as e:
        app.logger.error(f"Error setting up knowledgebase LLM: {e}")

# Setup all required components
def setup():
    setup_chatbot_llm()
    setup_knowledgebase_llm()

# Answer from knowledgebase
def answer_from_knowledgebase(message):
    global knowledgebase_qa
    if not knowledgebase_qa:
        app.logger.error("Knowledgebase LLM not initialized.")
        return "Error: Knowledgebase not available"
    
    app.logger.debug(f"Querying knowledgebase for: {message}")
    try:
        res = knowledgebase_qa({"query": message})
        app.logger.debug("Query successful")
        return res['result']
    except Exception as e:
        app.logger.error(f"Error querying knowledgebase: {e}")
        return "Error querying knowledgebase"

# Search knowledgebase
def search_knowledgebase(message):
    global knowledgebase_qa
    if not knowledgebase_qa:
        app.logger.error("Knowledgebase LLM not initialized.")
        return "Error: Knowledgebase not available"
    
    app.logger.debug(f"Searching knowledgebase for: {message}")
    try:
        res = knowledgebase_qa({"query": message})
        sources = ""
        for count, source in enumerate(res['source_documents'], 1):
            sources += f"Source {count}\n"
            sources += source.page_content + "\n"
        return sources
    except Exception as e:
        app.logger.error(f"Error searching knowledgebase: {e}")
        return "Error searching knowledgebase"

# Answer as chatbot
def answer_as_chatbot(message):
    global chatbot_llm_chain
    if not chatbot_llm_chain:
        app.logger.error("Chatbot LLM not initialized.")
        return "Error: Chatbot not available"
    
    app.logger.debug(f"Generating chatbot response for: {message}")
    try:
        res = chatbot_llm_chain.run(message)
        app.logger.debug(f"Generated response: {res}")
        return res
    except Exception as e:
        app.logger.error(f"Error generating chatbot response: {e}")
        return "Error generating chatbot response"

# Flask routes
@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json.get('message')
    response_message = answer_from_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/search', methods=['POST'])
def search():
    message = request.json.get('message')
    response_message = search_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json.get('message')
    response_message = answer_as_chatbot(message)
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="Chatbot and Knowledgebase")

if __name__ == "__main__":
    setup()  # Ensure setup is done before the app runs
    app.run(host='0.0.0.0', port=5001)
