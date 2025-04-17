# from src.chatbot import chat
# from src.source_locator import chat
from src.chatbotv2 import chat
import streamlit as st
import os

# Set Streamlit page settings
st.set_page_config(page_title="Orglens Chatbot", page_icon="📜")

st.title("📜 Orglens Chatbot! 🤖")
st.write("Ask me anything related to your organization!")

# Suggested questions dictionary
suggested_questions = {
    "Executive Summary": [
        "What is the gender distribution across the organization?",
        "How are employees distributed by tenure status?",
        "What is the breakdown of employees across hierarchy levels?",
        "Which departments have the highest number of employees?",
        "Which locations are most represented in the workforce?",
        "What are the primary legal entities represented in the organization?"
    ],
    "Department Snapshot": [
        "Who are the most frequently reached-out individuals within each department?",
        "What is the average degree of connectivity within departments?",
        "How frequently do employees connect within each department?",
        "What is the proportion of professional versus personal communication in departments?",
        "Are there individuals or teams within departments who appear disconnected?"
    ],
    "Hierarchy Level Snapshot": [
        "How does communication frequency vary across hierarchy levels?",
        "Which hierarchy levels show the strongest internal collaboration?",
        "Are there any levels where communication appears fragmented?",
        "Do professional reasons dominate over personal ones in each level's interactions?",
        "Are there levels with notably high or low mutual (reciprocal) connections?"
    ],
    "Inclusion Summary": [
        "Is gender representation equitable across the organization?",
        "Which departments show equitable inclusion based on gender?",
        "Are there any locations with potential gender-based bias?",
        "How does inclusion differ across legal entities?",
        "Are personal and professional networks equally inclusive?"
    ],
    "Legal Entity Snapshot": [
        "Which individuals are most connected within each legal entity?",
        "What is the average communication frequency within legal entities?",
        "How are professional and personal interactions distributed in each entity?",
        "Are there any isolated individuals or teams within legal entities?",
        "What percentage of communication is mutual in each entity?"
    ],
    "Location Snapshot": [
        "What is the average connectivity level across different office locations?",
        "How frequently do employees in each location connect with others?",
        "Are certain locations more professionally connected than others?",
        "Are there silos or disconnected individuals in specific locations?",
        "Who are the most connected individuals in each location?"
    ],
    "Engagement Summary": [
        "How many employees are highly engaged, and what defines them?",
        "What is the average number of connections among highly engaged employees?",
        "What types of communication (e.g., work, expertise, trust) are most common among engaged employees?",
        "Which roles or hierarchy levels are most engaged?",
        "What percentage of engaged employees have reciprocal relationships with their managers?"
    ],
    "Collaboration Overview": [
        "What is the volume of collaboration within versus between hierarchy levels?",
        "Which departments collaborate most effectively with others?",
        "Are there specific locations with stronger internal collaboration?",
        "Is collaboration primarily for professional or personal reasons?",
        "Which groups show no or very limited collaboration?"
    ],
    "Organizational Network Summary": [
        "Who are the top individuals others reach out to across the organization?",
        "What is the average in-degree and out-degree in the organizational network?",
        "What is the typical communication frequency across the organization?",
        "What proportion of connections are mutual across the network?",
        "Are there employees with no connections at all?"
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