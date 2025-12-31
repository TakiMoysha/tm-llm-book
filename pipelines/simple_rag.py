import argparse
from pathlib import Path
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.llms import HuggingFaceHub
import httpx
from langchain.llms.base import LLM

from langchain.chains import RetrievalQA

import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
# ==========================================================================================


class LMStudioLLM(LLM):
    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        response = httpx.post(
            "http://localhost:1234/v1/completions", json={"prompt": prompt, "max_tokens": 512, "stop": stop}
        )
        logging.info(response.json())
        return response.json()["choices"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "custom"


llm = LMStudioLLM()
# ==========================================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LANGUAGE_MODEL = ""


def vault_processing(path: Path):
    loader = DirectoryLoader(path, glob="*.md")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma.from_documents(docs, embeddings, persist_directory="tmp/simple_rag")
    db.persist()


def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory="tmp/simple_rag", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain


def ask_question(question: str):
    result = get_rag_chain()({"query": question})
    print("Answer:", result["result"])
    print("Sources:", result["source_documents"])
    for doc in result["source_documents"]:
        print(doc.metadata["source"], doc.page_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    operation_parser = parser.add_subparsers(dest="operations", help="Available operations")

    parser_process = operation_parser.add_parser("vault_processing", help="Embed documents from vault")
    parser_process.add_argument("--vault", type=str, help="path to documents vault", required=True)

    parser_ask = operation_parser.add_parser("ask", help="Ask a question")


    args = parser.parse_args()

    if args.command == "vault_processing":
        vault_processing(Path(args.vault))
    elif args.command == "ask_question":
        ask_question(args.question)
    else:
        parser.print_help()

