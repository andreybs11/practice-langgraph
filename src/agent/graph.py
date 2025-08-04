"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Any, Dict, TypedDict, Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


class State(TypedDict):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    # messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = [SystemMessage(content="You are a Principal Software Engineer that helps with software development.")]
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatGroq(model="llama3-8b-8192")
# llm = ChatGroq(model="llama-3.1-405b-reasoning", temperature=0)


def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]
    # configuration.get("my_configurable_param")

    system_prompt = SystemMessage(content=
        "You are a Software Engineer so try to make all responses technical."
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}


# Define the graph
builder = StateGraph(State, config_schema=Configuration)
builder.add_node(call_model)
builder.set_entry_point("call_model")
graph = builder.compile(name="Andrey Agent")
