from flask import Flask, render_template, request, jsonify, abort
import logging, os
from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
app.logger.setLevel(logging.DEBUG)

# Global LLM chains and knowledge base
chatbot_llm_chain = None
knowledgebase_qa = None

def setup_llms():
    """Set up the chatbot LLM and knowledge base QA."""
    global chatbot_llm_chain, knowledgebase_qa

    cohere_key = os.environ.get("COHERE_API_KEY")
    if not cohere_key:
        app.logger.error("Cohere API key not found.")
        return
    
    try:
        # Chatbot LLM setup
        template = "You are a chatbot. Previous conversation: {chat_history}. New question: {question}. Response:"
        chatbot_llm_chain = LLMChain(
            prompt=PromptTemplate(template=template, input_variables=["question", "chat_history"]),
            llm=Cohere(cohere_api_key=cohere_key),
            memory=ConversationBufferMemory(memory_key="chat_history")
        )

        # Knowledge base setup
        embeddings = CohereEmbeddings(cohere_api_key=cohere_key)
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        knowledgebase_qa = RetrievalQA.from_chain_type(
            llm=Cohere(cohere_api_key=cohere_key),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        app.logger.debug("LLMs initialized successfully.")
    
    except Exception as e:
        app.logger.error(f"Error initializing LLMs: {e}")

def query_knowledgebase(message):
    """Query the knowledge base."""
    if not knowledgebase_qa:
        abort(500, description="Knowledge base is not set up")
    try:
        res = knowledgebase_qa({"query": message})
        return res.get('result', 'No result'), res.get('source_documents', [])
    except Exception as e:
        app.logger.error(f"Error querying knowledge base: {e}")
        abort(500, description="Query failed")

def chatbot_response(message):
    """Generate response from the chatbot LLM."""
    if not chatbot_llm_chain:
        abort(500, description="Chatbot LLM is not set up")
    try:
        return chatbot_llm_chain.run(message)
    except Exception as e:
        app.logger.error(f"Error generating chatbot response: {e}")
        abort(500, description="Chatbot query failed")

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json.get('message', '')
    if not message:
        abort(400, description="Message is required")
    response_message, _ = query_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/search', methods=['POST'])
def search():
    message = request.json.get('message', '')
    if not message:
        abort(400, description="Message is required")
    _, sources = query_knowledgebase(message)
    source_texts = "\n".join([f"Source {i+1}\n{doc.page_content}" for i, doc in enumerate(sources)])
    return jsonify({'message': source_texts or "No sources available"}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json.get('message', '')
    if not message:
        abort(400, description="Message is required")
    response_message = chatbot_response(message)
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

if __name__ == "__main__":
    setup_llms()  # Initialize the LLMs
    app.run(host='0.0.0.0', port=5001)
