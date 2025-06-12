import streamlit as st
import openai
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

import os
from dotenv import load_dotenv
load_dotenv()
if not os.getenv("LANGCHAIN_API_KEY"):
    st.error("LANGCHAIN_API_KEY is not set!")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = 'Imans-Chatbot-Ollama-Models-For-Asking-Questions'


## prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful and informative assistant designed for question answering.
                    Please provide detailed and insightful responses to user queries.
                    If a user asks for help, offer a comprehensive overview of the chatbot's features, including:

                    * **How to ask questions:** Explain how users can phrase their questions effectively. For instance, mention using clear and concise language, and the types of information the bot is best equipped to handle.
                    * **Available commands:** If there are specific keywords or commands, list them and describe their functionality. This could be things like "help," "examples," or any other custom commands you might want to add.
                    * **Supported topics:** Briefly mention the chatbot's knowledge domain. Is it focused on a specific area like data science or general knowledge? This helps users to know what kind of questions to ask.
                    * **Examples:** Provide a few sample questions to demonstrate how to interact with the chatbot effectively.

                    Always be friendly, patient, and aim to provide the most useful information possible.""",),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, model, temperature, max_tokens):
    llm = OllamaLLM(model=model)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Imans-Chatbot-Ollama-Models-For-Asking-Questions")

## select the Ollama Model
llm= st.sidebar.selectbox("Select Open Source Model",["phi3:mini","gemma:2b","Moondream","tinyllama"])

## Adjust response parameter
##llm = Ollama(model=model_name,temperature=0.7....)
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


# Main interface for using input
st.write("🌈 I'm curious! What wonders do you have for me today? 🤔 Ask away! 🌟")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")
