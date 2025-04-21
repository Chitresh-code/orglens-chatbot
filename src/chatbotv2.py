from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from datetime import datetime
import os
from dotenv import load_dotenv

# ========= ENV & MODELS =========
load_dotenv()
openai_llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedding=embeddings)

# ========= LOAD DOCUMENTS =========
files = [
    "./data/department_snapshot.json",
    "./data/exec_summary_org.json",
    "./data/hierarchy_level_snapshot.json",
    "./data/legal_entity_snapshot.json",
    "./data/location_snapshot.json",
    "./data/org_engagement.json",
    "./data/overall_module_stats.json",
]
docs = []
for file in files:
    loader = JSONLoader(file_path=file, jq_schema=".", text_content=False)
    docs.extend(loader.load())
split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vector_store.add_documents(split_docs)

# ========= STATE =========
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rag_response: str
    neo4j_response: str
    final_response: str
    next: str
    current_query: str

# ========= RAG NODE =========
# rag_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are an AI assistant helping a high-level organizational leader, providing insights on organizational network data.\n"
#         "## Guidelines"
#             "Only answer ONA-related questions."
#             "Use available data without mentioning incompleteness."  
#             "Suggest a better question if no relevant data is found."
#             "\n"
#      "Respond based on the following context. Be concise and factual. Avoid stating uncertainty.\n\n"
#      "Context:\n{context}\n"),
#     ("user", "{query}")
# ])
# rag_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#         "# Role"
#             "You are ONA Insight Bot, an AI assistant providing insights on organizational networks, including communication patterns, collaboration, and engagement."
#             "\n"

#         "## Guidelines"
#             "Only answer ONA-related questions."
#             "Use available data without mentioning incompleteness."  
#             "Suggest a better question if no relevant data is found."
#             "\n"
            
#         "# Query Example:"
#             "Q:What percentage of employees are tenured vs. new hires?"
#             "A:Based on the organization's data:"
#                 "percentage of employees thar are tenured."
#                 "percentage that are new hires."
#             "\n"
            
#         "## Tool Guidelines"
#             "Use the `retrieve` tool to search for relevant information."
#             "Use the tool every time a query is received before responding."
#             "If the tool does not return relevant information, guide the user to ask a better question."
#             "Do not tell the user that more data is needed without calling the tool first."
#             "\n"

#         "# Conversation Guidelines"
#             "You are speaking with a high-ranking employee of an organization where we have conducted an ONA (Organizational Network Analysis)."
#             "The user is not an ONA specialist and does not have technical knowledge of the subject."
#             "Avoid mentioning data limitations or the need for more data. Simply present insights based on the available information without commenting on what is missing."
#             "Also do not mention that more data is required or that data is incomplete. Just present the data as it is and don't tell the user what is present or not just state facts."
#             "Do not refer to the data as `provided data.` Instead, use terms like `your company's data` or `according to your organization's insights` to maintain a personalized and authoritative tone."
#             "\n"
            
#         "# Notes"  
#             "Stay professional."  
#             "You can give response in multiple paragraphs and bullet points if needed."
#             "Present data directly without disclaimers, do not mention data incompleteness."
#             "Guide users toward better queries if needed. \n"  
#             "\nCurrent time: {time}.",
#         ),
#     ("user", "{query}")
# ])
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly and knowledgeable AI assistant supporting a high-level leader with insights from their organization's network data. "
     "Respond in a helpful, conversational, and confident tone — think of yourself as a smart, supportive colleague.\n\n"
     "## Tone Guidelines\n"
     "- Be warm, engaging, and professional.\n"
     "- Avoid sounding robotic or overly technical.\n"
     "- Imagine you’re chatting with an executive who’s short on time but appreciates a touch of human tone.\n\n"
     "## Response Guidelines\n"
     "- Use natural language, contractions (e.g., “you’re”, “it’s”), and light phrasing.\n"
     "- Only answer questions related to Organizational Network Analysis (ONA).\n"
     "- Base your responses on the data below without mentioning limitations or incompleteness.\n"
     "- If no relevant data is found, guide the user to ask a clearer or more helpful question (gently).\n\n"
     "Here's the relevant context from your company's data:\n{context}\n")
    ,
    ("user", "{query}")
])

def rag_fn(state: State):
    query = state["current_query"]
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = rag_prompt.invoke({"context": context, "query": query})
    response = openai_llm.invoke(prompt).content
    print("[RAG] Response:", response)
    return {"rag_response": response}

# ========= NEO4J NODE =========
# cypher_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="Translate this into Cypher:\n\n{question}"
# )
# cypher_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""Task: Generate a Cypher statement to query an organizational network analysis (ONA) graph database.

# Instructions:
#     - Use only the relationship types and properties provided in the schema below.
#     - Do not make up properties or relationships that are not explicitly listed.
#     - Only output a valid Cypher statement.
#     - Do not include explanations, apologies, or any non-Cypher text.
#     - Assume the graph is about people and their organizational relationships, attributes, and network connections.

# Schema:
# Node labels and properties:
#     - Person(id: STRING, name: STRING, last_name: STRING, value: STRING, first_name: STRING)
#     - Designation(value: STRING)
#     - Department(value: STRING) → ['Retail Training', 'Medical', 'People Function', 'Training Centre']
#     - Location(value: STRING)
#     - ReportingManager(value: STRING)
#     - JoiningDate(value: STRING)
#     - Email(value: STRING)
#     - LegalEntity(value: STRING) → ['Titan Company Limited', 'Titan Engineering and Automation Limited']
#     - Gender(type: STRING) → ['Male', 'Female']
#     - HierarchyLevel(level: INTEGER) → 1 to 10
#     - Leadership(type: STRING) → ['Non-Leader', 'Leader']
#     - Rating(value: INTEGER) → 1 to 10

# Relationships:
#     (:Person)-[:HAS_DESIGNATION]->(:Designation)
#     (:Person)-[:HAS_DEPARTMENT]->(:Department)
#     (:Person)-[:HAS_LOCATION]->(:Location)
#     (:Person)-[:HAS_REPORTINGMANAGER]->(:ReportingManager)
#     (:Person)-[:HAS_JOININGDATE]->(:JoiningDate)
#     (:Person)-[:HAS_EMAIL]->(:Email)
#     (:Person)-[:HAS_LEGALENTITY]->(:LegalEntity)
#     (:Person)-[:HAS_GENDER]->(:Gender)
#     (:Person)-[:HAS_HIERARCHY_LEVEL]->(:HierarchyLevel)
#     (:Person)-[:HAS_LEADERSHIP]->(:Leadership)
#     (:Person)-[:HAS_RATING]->(:Rating)
#     (:Person)-[:IS_CONNECTED]->(:Person)

# The question is:
# {question}
# """
# )
cypher_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You're a helpful AI assistant helping a leader explore their organization's network using Cypher queries over a graph database.

Goal:
Translate the user's natural language question into a Cypher query using ONLY the schema and relationship types provided below.

Rules:
- Use only the node labels and relationships listed.
- Do not invent or assume fields or connections that aren't in the schema.
- Only output a valid Cypher query. Do NOT add explanations or extra text.
- Write clean, readable Cypher — prefer clarity over complexity.
- Assume the graph represents people, their attributes, and organizational connections.

User’s Question:
{question}

Schema Reference:
Node Labels & Properties:
- Person(id: STRING, name: STRING, last_name: STRING, value: STRING, first_name: STRING)
- Designation(value: STRING)
- Department(value: STRING) → ['Retail Training', 'Medical', 'People Function', 'Training Centre']
- Location(value: STRING)
- ReportingManager(value: STRING)
- JoiningDate(value: STRING)
- Email(value: STRING)
- LegalEntity(value: STRING) → ['Titan Company Limited', 'Titan Engineering and Automation Limited']
- Gender(type: STRING) → ['Male', 'Female']
- HierarchyLevel(level: INTEGER) → 1 to 10
- Leadership(type: STRING) → ['Non-Leader', 'Leader']
- Rating(value: INTEGER) → 1 to 10

Relationships:
- (:Person)-[:HAS_DESIGNATION]->(:Designation)
- (:Person)-[:HAS_DEPARTMENT]->(:Department)
- (:Person)-[:HAS_LOCATION]->(:Location)
- (:Person)-[:HAS_REPORTINGMANAGER]->(:ReportingManager)
- (:Person)-[:HAS_JOININGDATE]->(:JoiningDate)
- (:Person)-[:HAS_EMAIL]->(:Email)
- (:Person)-[:HAS_LEGALENTITY]->(:LegalEntity)
- (:Person)-[:HAS_GENDER]->(:Gender)
- (:Person)-[:HAS_HIERARCHY_LEVEL]->(:HierarchyLevel)
- (:Person)-[:HAS_LEADERSHIP]->(:Leadership)
- (:Person)-[:HAS_RATING]->(:Rating)
- (:Person)-[:IS_CONNECTED]->(:Person)
"""
)

neo4j_graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=True
)
neo4j_chain = GraphCypherQAChain.from_llm(
    openai_llm,
    graph=neo4j_graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    allow_dangerous_requests=True,
    validate_cypher=True
)

def neo4j_fn(state: State):
    query = state["current_query"]
    try:
        result = neo4j_chain.invoke({"query": query})
        result = result.get("result")
    except Exception as e:
        result = f"Neo4j Error: {str(e)}"
    print("[NEO4J] Response:", result)
    return {"neo4j_response": result}

# ========= PRIMARY CONTROLLER NODE =========
# ========= LLM-BASED FALLBACK DECISION =========

# fallback_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an assistant that evaluates if a given AI response was helpful, grounded in context, and confident."),
#     ("user", "Response:\n{response}\n\nWas this response helpful and data-based for the Question:\n{query}?\n Answer ONLY 'YES' or 'NO'.")
# ])
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful AI reviewing an assistant's answer to see if it was clear, confident, and useful based on data."),
    ("user", "Here's what the assistant said:\n\n{response}\n\nWas it helpful, confident, and grounded in the data for the question:\n{query}?\n\nPlease answer ONLY 'YES' or 'NO'.")
])

def should_fallback_llm(response: str, query: str) -> bool:
    try:
        prompt = fallback_prompt.invoke({"response": response, "query": query})
        result = openai_llm.invoke(prompt).content.strip().lower()
        print("[FALLBACK LLM] Decision:", result)
        return result != "yes"
    except Exception as e:
        print("[FALLBACK LLM] Error:", e)
        return True  # fallback to neo4j on failure

def primary_controller(state: State):
    query = state["current_query"]
    rag_resp = state.get("rag_response", "")
    neo4j_resp = state.get("neo4j_response", "")

    if not rag_resp:
        print("[PRIMARY CONTROLLER] No RAG response yet, routing to RAG.")
        return {"next": "rag"}

    from_fallback = should_fallback_llm(rag_resp, query)
    print(f"[PRIMARY CONTROLLER] Fallback LLM decision: {from_fallback}")

    if from_fallback and not neo4j_resp:
        return {"next": "neo4j"}

    if not from_fallback:
        return {"next": "finalize_rag"}

    if neo4j_resp:
        return {"next": "finalize_neo4j"}

    return {"next": END}

# ========= FINALIZE NODES =========
# def finalize_rag(state: State):
#     return {"final_response": state["rag_response"]}
def finalize_rag(state: State):
    response = state["rag_response"]
    return {
        "final_response": f"{response}\n\nLet me know if you'd like to explore this further or dive into another area!"
    }

# def finalize_neo4j(state: State):
#     return {"final_response": state["neo4j_response"]}
def finalize_neo4j(state: State):
    response = state["neo4j_response"]
    return {
        "final_response": f"{response}\n\nHappy to help if you want to look into a different angle or clarify something!"
    }

# ========= GRAPH BUILD =========
memory = MemorySaver()
graph = StateGraph(State)
graph.add_node("controller", RunnableLambda(primary_controller))
graph.add_node("rag", RunnableLambda(rag_fn))
graph.add_node("neo4j", RunnableLambda(neo4j_fn))
graph.add_node("finalize_rag", RunnableLambda(finalize_rag))
graph.add_node("finalize_neo4j", RunnableLambda(finalize_neo4j))

graph.set_entry_point("controller")
graph.add_conditional_edges("controller", lambda state: state["next"])

graph.add_edge("rag", "controller")
graph.add_edge("neo4j", "controller")
graph.add_edge("finalize_rag", END)
graph.add_edge("finalize_neo4j", END)

compiled_graph = graph.compile(checkpointer=memory)

# ========= CHAT =========
def chat(query: str) -> str:
    config = {"configurable": {"thread_id": "session"}}
    events = compiled_graph.stream({
        "messages": [("user", query)],
        "current_query": query,
        "rag_response": "",
        "neo4j_response": "",
        "next": ""
    }, config=config, stream_mode="values")

    final = None
    for event in events:
        final = event
    return final.get("final_response", "No response generated.")