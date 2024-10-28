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
        knowledgebase_qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        print("Successfully setup the KB")
    except Exception as e:
        print("Error:", e)


def setup():
    setup_chatbot_llm()
    setup_knowledgebase_llm()

def answer_from_knowledgebase(message):
    global knowledgebase_qa
    app.logger.debug('Before query')
    res = knowledgebase_qa({"query": message})
    app.logger.debug('Query successful')

    return res['result']

def search_knowledgebase(message):
    global knowledgebase_qa
    res = knowledgebase_qa({"query": message})
    sources = ""
    for count, source in enumerate(res['source_documents'],1):
        sources += "Source " + str(count) + "\n"
        sources += source.page_content + "\n"
    return sources

def answer_as_chatbot(message):
    template = """Question: {question}
    Answer as if you are an expert Python developer"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    res = llm_chain.run(message)
    return res

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json['message']
    
    # Generate a response
    response_message = answer_from_knowledgebase(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200
    

@app.route('/search', methods=['POST'])
def search():    
    message = request.json['message']
    
    # Generate a response
    response_message = search_knowledgebase(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    
    # Generate a response
    response_message = answer_as_chatbot(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

if __name__ == "__main__":
    setup()
    app.run(host='0.0.0.0', port=5001)
