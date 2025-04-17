# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import JSONLoader
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langchain_core.messages import AIMessage, ToolMessage
# from langchain_core.vectorstores import InMemoryVectorStore
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import AnyMessage, add_messages
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import ToolNode
# from typing_extensions import TypedDict, Annotated
# from dotenv import load_dotenv
# from datetime import datetime
# from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
# import os

# # ========= ENVIRONMENT VARIABLES ==========
# load_dotenv()
# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# # ========== MODELS & VECTORS ==========
# openai_llm = ChatOpenAI(model="gpt-4o-mini")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vector_store = InMemoryVectorStore(embedding=embeddings)

# # ========== DOCUMENT LOADING ==========
# files = [
#     "./data/department_snapshot.json",
#     "./data/exec_summary_org.json",
#     "./data/hierarchy_level_snapshot.json",
#     "./data/legal_entity_snapshot.json",
#     "./data/location_snapshot.json",
#     "./data/org_engagement.json",
#     "./data/overall_module_stats.json",
# ]
# docs = []
# for file in files:
#     loader = JSONLoader(file_path=file, jq_schema=".", text_content=False)
#     docs.extend(loader.load())

# split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
# vector_store.add_documents(split_docs)

# # ========== STATE TYPE ==========
# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# # ========== PRIMARY ROUTER ==========
# # primary_prompt = ChatPromptTemplate.from_messages([
# #     ("system", """You are a routing assistant. You are NOT supposed to answer the question.
    
# # Your only job is to look at the user's query and decide which specialized assistant should handle it:

# # - If the question can be answered from internal documents or reports, respond with: RAG
# # - If the question is better suited to querying the Neo4j organizational graph, respond with: Neo4j

# # ❗IMPORTANT:
# # - ONLY respond with exactly one word: RAG or Neo4j
# # - DO NOT answer the question.
# # - DO NOT say anything else.
# # - DO NOT ask follow-up questions.

# # """),
# primary_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are the primary assistant. Based on the user's question, decide who should handle it:
# - The flow of answering the question is as follows:
#     1. First, give the question to "RAG" assistant.
#     2. If "RAG" cannot answer, then give the question to "Neo4j" assistant.
#     3. If "Neo4j" cannot answer, then suggest the user an improvement to the question.

# - Respond with ONLY one word: "RAG" if the answer should be found in internal reports/documents.
# - Or "Neo4j" if it requires querying the organizational graph database.

# - Do not answer the question yourself.
# - Do not provide any explanations or additional information.
# """),
#     ("user", "{query}")
# ])
# primary_assistant = primary_prompt | openai_llm

# def primary_router(state: State):
#     query = state["messages"][-1].content
#     decision = primary_assistant.invoke({"query": query}).content.strip().lower()
#     print(f"[ROUTER] Routing decision: {decision}")
#     return {"__routing__": decision}

# # ========== TOOL ERROR HANDLER ==========
# def handle_tool_error(state) -> dict:
#     error = state.get("error")
#     last_msg = state["messages"][-1]
#     tool_calls = getattr(last_msg, "tool_calls", [{"id": "unknown"}])

#     return {
#         "messages": [
#             ToolMessage(
#                 content=f"Error: {repr(error)}\nPlease fix your query or try rephrasing.",
#                 tool_call_id=tc["id"],
#             )
#             for tc in tool_calls
#         ]
#     }

# def create_tool_node_with_fallback(tools: list) -> ToolNode:
#     return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")

# # ========== RAG ASSISTANT ==========
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

# def rag_runnable_fn(state: State):
#     query = state["messages"][-1].content
#     print(f"[RAG] Received query: {query}")
#     docs = vector_store.similarity_search(query, k=3)

#     if not docs:
#         print("[RAG] No relevant documents found.")
#         return {"messages": AIMessage(content="Sorry, I couldn't find relevant information.")}

#     print(f"[RAG] Retrieved {len(docs)} documents.")
#     for i, doc in enumerate(docs):
#         print(f"[RAG] Doc {i+1} content (truncated): {doc.page_content[:200]}")

#     context = "\n\n".join(doc.page_content for doc in docs)
#     rag_prompt_with_context = rag_prompt.partial(context=context)
#     inputs = {"query": query, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#     print(f"[RAG] Prompt input keys: {list(inputs.keys())}, Context length: {len(context)} chars")

#     response = openai_llm.invoke(rag_prompt_with_context.invoke(inputs))
#     print(f"[RAG] Response: {response.content[:300]}")
#     return {"messages": AIMessage(content=response.content)}

# rag_assistant = RunnableLambda(rag_runnable_fn)

# # ========== NEO4J ASSISTANT ==========
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

# Examples:
# # List all people in 'Medical' department
# MATCH (p:Person)-[:HAS_DEPARTMENT]->(d:Department {value: \"Medical\"})
# RETURN p

# # Count how many leaders exist
# MATCH (p:Person)-[:HAS_LEADERSHIP]->(l:Leadership {type: \"Leader\"})
# RETURN count(p) AS leaderCount

# # Find direct connections of a person named \"Raj\"
# MATCH (p:Person {name: \"Raj\"})-[:IS_CONNECTED]->(conn:Person)
# RETURN conn

# The question is:
# {question}
# """
# )

# neo4j_graph = Neo4jGraph(
#     url=os.getenv("NEO4J_URI"),
#     username=os.getenv("NEO4J_USER"),
#     password=os.getenv("NEO4J_PASSWORD"),
#     enhanced_schema=True,
# )

# neo4j_chain = GraphCypherQAChain.from_llm(
#     openai_llm,
#     graph=neo4j_graph,
#     cypher_prompt=cypher_prompt,
#     verbose=True,
#     allow_dangerous_requests=True,
#     return_only_outputs=True,
# )

# def neo4j_runnable_fn(state: State):
#     try:
#         query = state["messages"][-1].content
#         print(f"[NEO4J] Received query: {query}")
#         result = neo4j_chain.invoke({"question": query})
#         print(f"[NEO4J] Result: {result[:300]}")
#         return {"messages": AIMessage(content=result)}
#     except Exception as e:
#         print(f"[NEO4J] Error: {e}")
#         return handle_tool_error({"messages": state["messages"], "error": e})

# neo4j_agent = RunnableLambda(neo4j_runnable_fn)

# # ========== GRAPH ==========
# memory = MemorySaver()
# builder = StateGraph(State)
# builder.add_node("primary", RunnableLambda(primary_router))
# builder.add_node("rag", rag_assistant)
# builder.add_node("neo4j", neo4j_agent)

# def route_from_primary(state: dict) -> str:
#     decision = state.get("__routing__", "")
#     return "neo4j" if "neo4j" in decision else "rag"

# builder.set_entry_point("primary")
# builder.add_conditional_edges("primary", route_from_primary)
# builder.add_edge("rag", END)
# builder.add_edge("neo4j", END)

# graph = builder.compile(checkpointer=memory)

# # ========== CHAT ==========
# def chat(query: str) -> str:
#     thread_id = "multi-agent-debug-thread"
#     config = {"configurable": {"thread_id": thread_id}}
#     events = graph.stream({"messages": ("user", query)}, config, stream_mode="values")
#     final = None
#     for event in events:
#         final = event
#     msg = final.get("messages")
#     return msg[-1].content if isinstance(msg, list) else msg.content

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
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI assistant helping a high-level organizational leader, providing insights on organizational network data.\n"
        "## Guidelines"
            "Only answer ONA-related questions."
            "Use available data without mentioning incompleteness."  
            "Suggest a better question if no relevant data is found."
            "\n"
     "Respond based on the following context. Be concise and factual. Avoid stating uncertainty.\n\n"
     "Context:\n{context}\n"),
    ("user", "{query}")
])
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
cypher_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Task: Generate a Cypher statement to query an organizational network analysis (ONA) graph database.

Instructions:
    - Use only the relationship types and properties provided in the schema below.
    - Do not make up properties or relationships that are not explicitly listed.
    - Only output a valid Cypher statement.
    - Do not include explanations, apologies, or any non-Cypher text.
    - Assume the graph is about people and their organizational relationships, attributes, and network connections.

Schema:
Node labels and properties:
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
    (:Person)-[:HAS_DESIGNATION]->(:Designation)
    (:Person)-[:HAS_DEPARTMENT]->(:Department)
    (:Person)-[:HAS_LOCATION]->(:Location)
    (:Person)-[:HAS_REPORTINGMANAGER]->(:ReportingManager)
    (:Person)-[:HAS_JOININGDATE]->(:JoiningDate)
    (:Person)-[:HAS_EMAIL]->(:Email)
    (:Person)-[:HAS_LEGALENTITY]->(:LegalEntity)
    (:Person)-[:HAS_GENDER]->(:Gender)
    (:Person)-[:HAS_HIERARCHY_LEVEL]->(:HierarchyLevel)
    (:Person)-[:HAS_LEADERSHIP]->(:Leadership)
    (:Person)-[:HAS_RATING]->(:Rating)
    (:Person)-[:IS_CONNECTED]->(:Person)

The question is:
{question}
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

fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that evaluates if a given AI response was helpful, grounded in context, and confident."),
    ("user", "Response:\n{response}\n\nWas this response helpful and data-based for the Question:\n{query}?\n Answer ONLY 'YES' or 'NO'.")
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
def finalize_rag(state: State):
    return {"final_response": state["rag_response"]}

def finalize_neo4j(state: State):
    return {"final_response": state["neo4j_response"]}

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