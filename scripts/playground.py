import logging
import os

import pytest
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from qdrant_client.grpc import VectorParams
from qdrant_client.models import Distance

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

OPENAI_LOCAL_URL = "http://127.0.0.1:1234/v1/"


class UseCase:
    pass


class UCChatSummarize(UseCase):
    llm: ChatOpenAI
    prompt_template = """
    %INSTRUCTIONS:
    Please summarize the following piece of text.
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


def summary_of_web_page():
    uc = UCChatSummarize(llm)
    input = uc.prompt.format(user_input="")
    result = uc.llm.invoke(input)

    print(result)


# ==========================================================================================
# make language kernel, up to 2000 tokens (1500 words);
# RAG for rare and special cases;
# pipeline for startup;


EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
EMBEDDING_MODEL_DIMENSIONS = 768


def prepare_rag_documents():
    rag_doc = UnstructuredMarkdownLoader("./ork_speech.rag.ignore.md", mode="elements").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_rag_document = text_splitter.split_documents(rag_doc)
    num_total_characters = sum([len(x.page_content) for x in split_rag_document])
    logging.debug(f"D.{len(split_rag_document)}: avr. {num_total_characters / len(split_rag_document):,.0f} chars.")
    logging.debug(f"Total: {num_total_characters:,} chars")
    return split_rag_document


def get_roleplay_char_pipeline(llm: ChatOpenAI):
    document = prepare_rag_documents()
    pass


from langchain.chains import RetrievalQA


@pytest.mark.target
def test_embedding_text():
    documents = prepare_rag_documents()
    # check_embedding_ctx_length=False - required for LMStudio
    embeddings = OpenAIEmbeddings(base_url=OPENAI_LOCAL_URL, check_embedding_ctx_length=False)
    # vectorestores, working with OpenAI api
    vs = get_vectorestore(embedding=embeddings)
    docsearch = QdrantVectorStore.from_documents(
        documents,
        embeddings,
        location=":memory:",
    )
    # retrieval engine
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     embeddings=embeddings,
    # )
    print(vs)


# ==========================================================================================
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types


def get_vectorestore(*, embedding: OpenAIEmbeddings | None = None):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="ork_speech_embeddings",
        vectors_config=types.VectorParams(size=EMBEDDING_MODEL_DIMENSIONS, distance=Distance.COSINE),
    )
    return QdrantVectorStore(
        client=client,
        collection_name="ork_speech_embeddings",
        embedding=embedding,
    )


@pytest.mark.integration
def test_vectorstore():
    embeddings = OpenAIEmbeddings(base_url=OPENAI_LOCAL_URL, check_embedding_ctx_length=False)
    vs = get_vectorestore(embedding=embeddings)


# ==========================================================================================

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True, scope="package")
def openai_api_key():
    if (key := os.getenv("OPENAI_API_KEY")) is None:
        pytest.fail("OPENAI_API_KEY is not set")

    return SecretStr(key)


@pytest.fixture
def llm(openai_api_key) -> ChatOpenAI:
    llm = ChatOpenAI(
        temperature=0.8,
        api_key=openai_api_key,
        max_retries=3,
        base_url="https://api.poe.com/v1",
        model="Assistant",
    )
    return llm


async def test_summarize(llm: ChatOpenAI):
    llm.temperature = 0
    summarize(llm)
