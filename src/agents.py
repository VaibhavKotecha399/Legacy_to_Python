from typing import Annotated
from langchain.messages import SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

LLM = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=1000)

def get_code_sample(state : State):
    """This function will get the file from folder in txt format and return the content"""
    all_docs=[]
    pdf_directories= Path("./Sample_JCLs_&_Cobol/")
    pdf_files= list(pdf_directories.glob("**/*.txt"))
    print(f"DEBUG: Found {len(pdf_files)} text files.")
    for pdf_file in pdf_files:
        with open(pdf_file, "r") as f:
            all_docs.append(f.read())
    context = SystemMessage(content="You are a helpful assistant that extracts information from code samples.")         
    return {"messages": [context] + all_docs}

def analyze_code_sample(state : State):
    """This function will analyze the code sample using LLM and return the response"""
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

graph_builder.add_edge(START, "CodeParser")
graph_builder.add_node("CodeParser", get_code_sample)
graph_builder.add_edge("CodeParser","LLMAnalyzer")
graph_builder.add_node("LLMAnalyzer",analyze_code_sample)
graph_builder.add_edge("CodeParser", END)

graph=graph_builder.compile()

response = graph.invoke({'messages':"Convert this code to Python code?"})
response["messages"][-1].content