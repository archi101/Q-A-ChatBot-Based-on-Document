from langchain.embeddings import SentenceTransformerEmbeddings
import google.generativeai as genai
import streamlit as st
import os
from langchain.vectorstores import Chroma
os.environ['Google_API_KEY'] ="AIzaSyDXb4x7Ywl79ND7l_MY18OgJJZKkVDXEAg"
import pickle

genai.configure(api_key=os.environ['Google_API_KEY'])
model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 
def setup():
    embedding = pickle.load(open('embedding.pkl','rb'))
    document = pickle.load(open('document.pkl','rb'))
    db = Chroma.from_documents(document, embedding)
    return db
def find_match(input):
    # input_em = model.encode(input).tolist()
    # result = index.query(input_em, top_k=2, includeMetadata=True)
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
    db=setup()
    matching_doc=db.similarity_search_with_score(input,k=2)
    return matching_doc[0][0].page_content+matching_doc[1][0].page_content


def query_refiner(conversation, query):

    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    # "response_mime_type": "text/plain",
     }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=f"Your name is Sam, given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\\n\\nCONVERSATION LOG: \\n{conversation}\\n\\nQuery: {query}\\n\\nRefined Query:",
    )
    # refined_query=model.generate_content(conversation=conversation, query=query)
    # return refined_query
    chat_session = model.start_chat(
    history=[])
    response=chat_session.send_message("Conversation Log:{conversation} Query:{query}")
    return response.text

def get_conversation_string():

    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string