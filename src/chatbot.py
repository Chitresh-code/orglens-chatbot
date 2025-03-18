from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langchain_core.tools import tool
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated
import os

load_dotenv()

# Set the GOOGLE_API_KEY environment variable if not already set
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = InMemoryVectorStore(embedding=embeddings)

files = [
        r"./data/department_snapshot.json",
        r"./data/exec_summary_org.json",
        r"./data/hierarchy_level_snapshot.json",
        r"./data/legal_entity_snapshot.json",
        r"./data/location_snapshot.json",
        r"./data/org_engagement.json",
        r"./data/overall_module_stats.json",
    ]

docs = []

for file in files:
    loader = JSONLoader(
        file_path=file,
        jq_schema=".",  # Extracts the entire JSON object
        text_content=False  # Prevents expecting page_content as a string
    )
    data = loader.load()
    docs.extend(data)
    
print(f"Loaded {len(docs)} documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

print(f"Split {len(split_docs)} documents.")

result = vector_store.add_documents(documents=split_docs)

print(f"Added {len(result)} documents to the vector store.")

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
    
# NOTE: this function is for printing the messages in the console only for debugging purposes
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
    
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
    
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            # user_id = configuration.get("user_id", None)
            # state = {**state, "user_info": user_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "# Role"
            "You are ONA Insight Bot, an AI assistant providing insights on organizational networks, including communication patterns, collaboration, and engagement."
            "\n"

            "## Guidelines"
            "Only answer ONA-related questions."
            "Use available data without mentioning incompleteness."  
            "Suggest a better question if no relevant data is found."
            "\n"
            
            "# Query Example:"
            "Q:What percentage of employees are tenured vs. new hires?"
            "A:Based on the organization's data:"
                "percentage of employees thar are tenured."
                "percentage that are new hires."
            "\n"

            "# Conversation Guidelines"
            "You are speaking with a high-ranking employee of an organization where we have conducted an ONA (Organizational Network Analysis)."
            "The user is not an ONA specialist and does not have technical knowledge of the subject."
            "Avoid mentioning data limitations or the need for more data. Simply present insights based on the available information without commenting on what is missing."
            "Also do not mention that more data is required or that data is incomplete. Just present the data as it is and don't tell the user what is present or not just state facts."
            "Do not refer to the data as `provided data.` Instead, use terms like `your company's data` or `according to your organization's insights` to maintain a personalized and authoritative tone."
            
            "# Notes"  
            "Stay professional."  
            "You can give response in multiple paragraphs and bullet points if needed."
            "Present data directly without disclaimers, do not mention data incompleteness."
            "Guide users toward better queries if needed. \n"  
            # "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

assistant_tools = [
    retrieve
]
assistant_runnable = assistant_prompt | llm.bind_tools(assistant_tools)

builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(assistant_tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def chat(query: str) -> str:
    thread_id = "abc"
    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }
    
    _printed = set()

    events = graph.stream(
        {"messages": ("user", query)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
        
    message = event.get("messages")  # Retrieve messages from event
    if message:  # Check if messages exist
        if isinstance(message, list):  # If messages is a list, get the last one
            return message[-1].content
        return message.content
    return "No response found."