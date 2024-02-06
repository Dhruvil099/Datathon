import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import streamlit as st
from langchain.agents.agent_types import AgentType
from streamlit_chat import message
import os

# # Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = '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'

# # Initialize session state variables for storing prompts and responses
# if 'prompts' not in st.session_state:
#     st.session_state.prompts = []
# if 'responses' not in st.session_state:
#     st.session_state.responses = []

# # Function to handle sending messages
# def send_click():
#     if st.session_state.user != '':
#         prompt = st.session_state.user
#         response = agent.run(prompt)

#         st.session_state.prompts.append(prompt)
#         st.session_state.responses.append(response)

# # Main Streamlit application
# st.title('My Data Analysis Chatbot')

# # File uploader for CSV files
# uploaded_file = st.file_uploader("Mumbai event data Feb 01.csv", type='csv')

# # Read the CSV file into a pandas dataframe
# if uploaded_file is not None:
#     csv_data = uploaded_file.read()
#     with open(uploaded_file.name, 'wb') as f:
#         f.write(csv_data)

#     df = pd.read_csv(uploaded_file.name)
#     st.dataframe(df.head())

#     # Create a LangChain chat model and agent
#     chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
#     agent = create_pandas_dataframe_agent(chat, df, verbose=True)

#     # Text input and button for user interaction
#     st.text_input("Ask something:", key="user")
#     st.button("Send", on_click=send_click)

#     # Display chat history
#     if st.session_state.prompts:
#         for i in range(len(st.session_state.responses) -  1, -1, -1):
#             message(st.session_state.responses[i], key=str(i), seed='Milo')
#             message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)

# # # Run the Streamlit app
# # !python -m streamlit run chatbot.py


import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

# Initialize session state variables for storing prompts and responses
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Function to handle sending messages


def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        response = agent.run(prompt)

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)


# Main Streamlit application
st.title('My Data Analysis Chatbot')

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type='csv')

# Read the CSV file into a pandas dataframe
if uploaded_file is not None:
    csv_data = uploaded_file.read()
    with open(uploaded_file.name, 'wb') as f:
        f.write(csv_data)

    df = pd.read_csv(uploaded_file.name)
    st.dataframe(df.head())

    # Create a LangChain chat model and agent
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    agent = create_pandas_dataframe_agent(chat, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)

    # Text input and button for user interaction
    st.text_input("Ask something:", key="user")
    st.button("Send", on_click=send_click)

    # Display chat history
    if st.session_state.prompts:
        for i in range(len(st.session_state.responses) - 1, -1, -1):
            st.write(st.session_state.responses[i])
            st.write(st.session_state.prompts[i])
