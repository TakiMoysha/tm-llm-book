from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI


llm: ChatOpenAI = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    temperature=0,
)


@tool
def translate_orks(word: str) -> str:
    """Returns the length of a word."""
    print("Using tool")
    return "DA LOWDIZT ORK IZ DA BESTEST ORK"


tools = [translate_orks]
llm_with_tools = llm.bind_tools(tools=tools)

# ==========================================================================================


def test_local_question():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, but don't know current events",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    list(agent_executor.stream({"input": "lets test target function with adsfjALJhfasdj argument"}))
