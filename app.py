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

def setup_chatbot_llm():
    global chatbot_llm_chain
    template = """
    You are a chatbot that had a conversation with a human. Consider the previous conversation to answer the new question.

    Previous conversation:{chat_history}
    New human question: {question}

    Response:"""

    prompt = PromptTemplate(template=template, input_variables=["question", "chat_history"])
    llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    memory = ConversationBufferMemory(memory_key="chat_history")
    chatbot_llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, memory=memory)
    

def setup_knowledgebase_llm():
    global knowledgebase_qa
    app.logger.debug('Setting KB')
    try:
        # Used for mathematical rep of the book
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        # db has the textual representation of the book.
        # we are loading and converting that to a mathematical model and persisting it in the chromaDB
        # and creating a search index
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        if vectordb is None:
            app.logger.error("Chroma vector store initialization failed.")
        knowledgebase_qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        app.logger.debug("Successfully setup the KB")
    except Exception as e:
        app.logger.error(f"Knowledgebase setup error: {str(e)}")
        raise e


def setup():
    setup_chatbot_llm()
    setup_knowledgebase_llm()

def answer_from_knowledgebase(message):
    global knowledgebase_qa
    app.logger.debug('Before query')
    try:
        res = knowledgebase_qa({"query": message})
        app.logger.debug('Query successful')
        if 'result' in res:
            return res['result']
        else:
            return "No answer found in the knowledge base."
    except Exception as e:
        app.logger.error(f"Error querying knowledge base: {str(e)}")
        return "Sorry, I couldn't retrieve the answer from the knowledge base."

def search_knowledgebase(message):
    global knowledgebase_qa
    try:
        res = knowledgebase_qa({"query": message})
        sources = ""
        for count, source in enumerate(res['source_documents'], 1):
            sources += "Source " + str(count) + "\n"
            sources += source.page_content + "\n"
        return sources
    except Exception as e:
        app.logger.error(f"Error searching knowledge base: {str(e)}")
        return "Sorry, I couldn't complete the search in the knowledge base."

def answer_as_chatbot(message):
    template = """Question: {question}
    Answer as if you are an expert Python developer"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    try:
        res = llm_chain.run(message)
        return res
    except Exception as e:
        app.logger.error(f"Error answering as chatbot: {str(e)}")
        return "Sorry, I couldn't generate a chatbot answer."


@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json.get('message', None)
    if not message:
        app.logger.error("No message received in request.")
        abort(400, "No message provided")

    response_message = answer_from_knowledgebase(message)
    app.logger.debug(f"Response from KB: {response_message}")

    return jsonify({'message': response_message}), 200
    

@app.route('/search', methods=['POST'])
def search():    
    message = request.json.get('message', None)
    if not message:
        app.logger.error("No message received in request.")
        abort(400, "No message provided")
    
    # Generate a response
    response_message = search_knowledgebase(message)
    app.logger.debug(f"Response from Search: {response_message}")

    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json.get('message', None)
    if not message:
        app.logger.error("No message received in request.")
        abort(400, "No message provided")

    # Generate a response
    response_message = answer_as_chatbot(message)
    app.logger.debug(f"Response from Chatbot: {response_message}")

    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

if __name__ == "__main__":
    setup()
    app.run(host='0.0.0.0', port=5001)
