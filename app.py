# from src.chatbot import chat
from src.source_locator import chat
import streamlit as st
import os

# Set Streamlit page settings
st.set_page_config(page_title="Orglens Chatbot", page_icon="📜")

st.title("📜 Orglens Chatbot! 🤖")
st.write("Ask me anything related to your organization!")

# Suggested questions dictionary
suggested_questions = {
    "Executive Summary": [
        "What are the key insights and strategic takeaways from the overall network analysis?"
    ],
    "Network Structure": [
        "How is communication distributed across the organization, and who holds central positions in the network?"
    ],
    "Collaboration Insights": [
        "Which departments or roles are collaborating effectively, and where are the communication gaps?"
    ],
    "Leadership Recommendations": [
        "What steps can leadership take to improve knowledge flow and cross-functional collaboration?"
    ],
    "Action Plan": [
        "What are the recommended next steps to enhance organizational connectivity over time?"
    ],
    "Appendix & Methodology": [
        "How was the data collected and what analytical methods were used to generate these insights?"
    ]
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

# Display chat history
for message in st.session_state.messages:
    role, text = message
    st.chat_message(role).write(text)

# Category selection logic
st.write("💡 **Select a Category to Explore Suggested Questions:**")

# Display category buttons
category_cols = st.columns(3)
for i, category in enumerate(suggested_questions.keys()):
    if category_cols[i % 3].button(category):
        st.session_state.selected_category = category

# If a category was selected, show its questions
if st.session_state.selected_category:
    st.markdown(f"### ✳️ Questions from: *{st.session_state.selected_category}*")
    question_list = suggested_questions[st.session_state.selected_category]
    for q in question_list:
        if st.button(q):
            st.session_state.selected_question = q
            st.session_state.selected_category = None  # Optional: reset after selection
            st.rerun()

# Handle user input
user_query = st.chat_input("Type your question here...")

# If a question was selected from the suggestions
if st.session_state.selected_question:
    user_query = st.session_state.selected_question
    st.session_state.selected_question = None  # Reset after processing

if user_query:
    st.session_state.messages.append(("user", user_query))
    answer = chat(user_query)
    st.session_state.messages.append(("ai", answer))

    st.chat_message("user").write(user_query)
    st.chat_message("ai").write(answer)