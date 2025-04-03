from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import date, datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
import os

load_dotenv()

# Set the GOOGLE_API_KEY environment variable if not already set
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = InMemoryVectorStore(embedding=embeddings)

# Step 1: Load the HTML
def html_loader(filepath: str) -> Document:
    with open(filepath, "r") as f:
        html_string = f.read()
    return Document(page_content=html_string, metadata={"source": filepath})

# Step 2: Split the documents in a structured way
def split_html_sections(html_str: str) -> list[Document]:
    soup = BeautifulSoup(html_str, "html.parser")
    section_divs = soup.find_all("div", class_="section")
    
    documents = []

    for div in section_divs:
        header = div.find("h1")
        paragraphs = div.find_all("p")

        if header:
            title = header.get_text()
            for idx, p in enumerate(paragraphs):
                content = p.get_text()
                metadata = {
                    "source": title,
                    "seq_num": idx
                }
                documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

doc = html_loader(r"./data/index.html")
split_docs = split_html_sections(doc.page_content)

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

@tool
def generate_link(title: str):
    """
    Generate a link to the website from the title provided
    NOTE: It can only generate a valid link if the title is provided from the source and not any random word

    Args:
     - title: Title of the page you want the link for

    Returns:
     - Link to the website
    """
    title = title.strip().lower().replace(" ", "-")
    link = f"http://127.0.0.1:3000/backend/data/index.html#{title}"
    return link

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
            "You are a bot designed to identify which section of the website contains the answer to the user's query"
            "\n"

            "## Guidelines"
            "Never answer the query"
            "Only Identify the source of the query and tell the user that source and provide link"
            "If the query is not answered by any source then apologize and say you cannot answer"
            "Do not answer the question by yourself call the tools to help you."
            "\n"

            "# Process to answer a query"
            "Step 1: Call the `retrieve` tool to retrieve the data from relevant documents."
            "The `retrieve` tool will return the source and the content of the document."
            "You can use the content to understand if the query is being answered or not by the source."
            "Calling the `retrieve tool after the query is the most important step. Do not skip this step."
            "Without this step you will not be able to identify the source of the query."
            "Step 2: Figure out from which source the query is being answered. If the query is not being answered by the source then tell the user you can't answer"
            "Step 3: Generate a link using `generate_link` tool for the source title provided."
            "Make sure to call the `generate_link` tool after the `retrieve` tool."
            "The `generate_link` tool will return the link to the source."
            "Do not skip this step."
            "Step 4: Answer to the user with the section and the link."
            "\n"

            "# Query Example:"
            "Q:Which employee has the higest degree centrality?"
            "A: Your question can be answered in the Network Struction section.\nHere's a link to the section: <insert-link>"
            "\n"

            "# Notes"
            "Stay professional."
            "You can give response in bullet points if needed."
            "Present data directly without disclaimers, do not mention data incompleteness.\n"
            # "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

assistant_tools = [
    retrieve, generate_link
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