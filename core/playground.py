from functools import lru_cache
import logging
import os

from lmstudio._sdk_models import GpuSetting
import pytest
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import Document, HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from qdrant_client.grpc import VectorParams
from qdrant_client.models import Distance

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

OPENAI_POE_URL = "https://api.poe.com/v1"
OPENAI_LOCAL_URL = "http://127.0.0.1:1234/v1/"
LLM_MODEL = "mistralai/mistral-nemo-instruct-2407"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
EMBEDDING_MODEL_DIMENSIONS = 768


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
    print("R", result)
    print("R", f"{len(input)} -> {len(result.content)}")


def summary_of_web_page():
    uc = UCChatSummarize(llm)
    input = uc.prompt.format(user_input="")
    result = uc.llm.invoke(input)

    print("R", result)


# ==========================================================================================
# make language kernel, up to 2000 tokens (1500 words);
# RAG for rare and special cases;
# pipeline for startup;


from langchain.chains.retrieval_qa.base import RetrievalQA


def prepare_rag_documents() -> list[Document]:
    rag_doc = UnstructuredMarkdownLoader("./data/ork_speech.rag.ignore.md", mode="elements").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_rag_document = text_splitter.split_documents(rag_doc)
    num_total_characters = sum([len(x.page_content) for x in split_rag_document])
    logging.debug(f"D.{len(split_rag_document)}: avr. {num_total_characters / len(split_rag_document):,.0f} chars.")
    logging.debug(f"Total: {num_total_characters:,} chars")
    return split_rag_document


def load_qa_chain_with_docs(llm: ChatOpenAI, documents: list[Document]):
    embeddings = get_lms_embedding_model()
    # vectorestores, working with OpenAI api
    vs = get_vectorestore(
        embeddings,
        EMBEDDING_MODEL_DIMENSIONS,
    )
    docsearch = vs.from_documents(
        documents,
        embeddings,
        location=":memory:",
    )
    # retrieval engine
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )


@pytest.mark.cookbook
def test_qa_orks_glossary(llm):
    rag_ork = prepare_rag_documents()
    qa = load_qa_chain_with_docs(llm, rag_ork)
    query = "How to translate the word 'weapon', give an example."
    logging.info(qa.run(query))


# ==========================================================================================
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate

# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


@pytest.mark.cookbook
def test_extraction_data(llm):
    """Try out structured output parser."""

    instruction = """
    You will be given a sentence with a fruits of names, extract those fruit names and assign an emoji to them.
    Return the frouit name and emojis in a python dict.
    """
    fruit_names = """
    Apple, Pear, this is an kiwi
    """

    prompt = instruction + fruit_names

    response_schemas = [
        ResponseSchema(name="name", description="The name of the fruit"),
        ResponseSchema(name="emoji", description="Emoji associated with the fruit"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "Given a command from the user, extract the fruits names and assign an emoji to them. \n\n{format_instructions}. \n{user_prompt}"
            ),
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions},
    )

    fruit_query = prompt.format_prompt(user_prompt="I really like apples and pears")
    logging.info(llm.invoke(fruit_query).content)


# ==========================================================================================
from langchain.evaluation.qa import QAEvalChain


def load_qa_eval_chain_with_docs(llm: ChatOpenAI, documents: list[Document]):
    embeddings = get_lms_embedding_model()
    # vectorestores, working with OpenAI api
    vs = get_vectorestore(embeddings, EMBEDDING_MODEL_DIMENSIONS)
    docsearch = vs.from_documents(documents, embeddings, location=":memory:")

    return QAEvalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        input_key="question",
    )


@pytest.mark.target
def test_evaluation(llm):
    """Try out evaluation -"""
    rag_ork = prepare_rag_documents()
    chain = load_qa_eval_chain_with_docs(llm, rag_ork)

    question_answers = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        # {"question": "What does it mean when the leader shout?", "answer": ""},
        # {"question": "Number after ten", "answer": "lots"},
    ]
    predictions = chain.apply(question_answers)
    logging.info(
        chain.evaluate(
            question_answers,
            predictions,
            question_key="question",
            prediction_key="result",
            answer_key="answer",
        )
    )


# ==========================================================================================
import lmstudio as lms
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types


@lru_cache
def get_lms_embedding_model():
    embedding_model = lms.list_loaded_models("embedding")
    if embedding_model is None or len(embedding_model) == 0:
        embedding_model = lms.embedding_model(EMBEDDING_MODEL)
        logging.info(embedding_model)

    # check_embedding_ctx_length=False - required for LMStudio
    return OpenAIEmbeddings(base_url=OPENAI_LOCAL_URL, check_embedding_ctx_length=False)


def get_vectorestore(embedding: OpenAIEmbeddings, embedding_dimensions):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="ork_speech_embeddings",
        vectors_config=types.VectorParams(
            size=embedding_dimensions,
            distance=Distance.COSINE,
        ),
    )
    return QdrantVectorStore(
        client=client,
        collection_name="ork_speech_embeddings",
        embedding=embedding,
    )


@pytest.mark.integration
def test_vectorstore_with_embeddings():
    embedding = get_lms_embedding_model()
    vs = get_vectorestore(embedding, EMBEDDING_MODEL_DIMENSIONS)


# ==========================================================================================

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True, scope="package")
def openai_api_key():
    if (key := os.getenv("OPENAI_API_KEY")) is None:
        pytest.fail("OPENAI_API_KEY is not set")

    return SecretStr(key)


@pytest.fixture(name="llm")
def _llm(openai_api_key) -> ChatOpenAI:
    """Try load llm from lmstudio."""
    loaded_llms = lms.list_loaded_models()

    if len(loaded_llms) == 0:
        lms.llm(
            LLM_MODEL,
            config=lms.LlmLoadModelConfig(
                context_length=8000,
            ),
        )

    llm = ChatOpenAI(
        temperature=0.5,
        api_key=openai_api_key,
        max_retries=3,
        base_url=OPENAI_LOCAL_URL,
        model="Assistant",
    )
    return llm


async def test_summarize(llm: ChatOpenAI):
    llm.temperature = 0
    summarize(llm)
