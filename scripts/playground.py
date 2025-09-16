import pytest
import os

from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import SecretStr


class UseCase:
    pass


class UCChatSummarize(UseCase):
    llm: ChatOpenAI
    prompt_template = """
    %INSTRUCTIONS:
    Please summarize the followind piece of text.
    Respond in a manner that a 5 year old would understand.

    %TEXT:
    {user_input}
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        super().__init__()

    @property
    def prompt(self):
        return ChatPromptTemplate.from_messages(
            messages=[
                ("user", self.prompt_template),
            ],
        )

    @property
    def chain(self):
        return self.prompt | self.llm


def summarize(llm):
    uc = UCChatSummarize(llm)
    input = uc.prompt.format(
        user_input="""
            For the next 130 years, debate raged.
            Some scientists called Prototaxites a lichen, others a fungus, and still others clung to the notion that it was some kind of tree.
            “The problem is that when you look up close at the anatomy, it’s evocative of a lot of different things, but it’s diagnostic of nothing,” says Boyce, an associate professor in geophysical sciences and the Committee on Evolutionary Biology.
            “And it’s so damn big that when whenever someone says it’s something, everyone else’s hackles get up: ‘How could you have a lichen 20 feet tall?’”
        """
    )
    result = uc.llm.invoke(input)
    print(result)
    print(f"{len(input)} -> {len(result.content)}")


# ==========================================================================================

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True, scope="package")
def openai_api_key():
    if (key := os.getenv("OPENAI_API_KEY")) is None:
        pytest.fail("OPENAI_API_KEY is not set")

    return SecretStr(key)


@pytest.fixture(autouse=True)
def llm(openai_api_key):
    return ChatOpenAI(
        temperature=0.8,
        api_key=openai_api_key,
        max_retries=3,
        base_url="https://api.poe.com/v1",
        model="Assistant",
    )


async def test_summarize(llm):
    summarize(llm)
