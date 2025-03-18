from src.chatbot import chat
import streamlit as st
import os

# Change page header to Orglens Chatbot
st.set_page_config(page_title="Orglens Chatbot", page_icon="📜")

# Streamlit UI
st.title("📜 Orglens Chatbot! 🤖")
st.write("Ask me anything related to your organization!")

# # Load data if not already loaded
# if "data_loaded" not in st.session_state:
#     st.session_state.data_loaded = False
    
# if not st.session_state.data_loaded:
#     load_data()
#     st.session_state.data_loaded = True

# Suggested questions
suggested_questions = [
    "What is the gender diversity ratio?",
    "What is the percentage of disconnected individuals in the organization?",
    "Give me a list of employees that are most reached out to",
    "What is the distribution of employees between departments",
    "What is the distribution of employees between different locations",
    "What is the employee engagement score?",
]

# Session state for memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# Display chat history
for message in st.session_state.messages:
    role, text = message
    st.chat_message(role).write(text)

# Display suggested questions as buttons
st.write("💡 **Suggested Questions:**")
cols = st.columns(len(suggested_questions))

for i, question in enumerate(suggested_questions):
    if cols[i].button(question):
        st.session_state.selected_question = question  # Store selected question

# Use selected question if available, else wait for user input
user_query = st.chat_input("Type your question here...")

# If a question was selected, process it immediately
if st.session_state.selected_question:
    user_query = st.session_state.selected_question
    st.session_state.selected_question = None  # Reset after processing

if user_query:
    st.session_state.messages.append(("user", user_query))

    # Get chatbot response
    answer = chat(user_query)

    st.session_state.messages.append(("ai", answer))

    # Display messages
    st.chat_message("user").write(user_query)
    st.chat_message("ai").write(answer)
