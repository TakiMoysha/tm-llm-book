import os
from pathlib import Path

MODEL_PATH = Path(os.getenv("STOREDIR", "")) / "ai-models/osmosis-ai/osmosis-mcp-4b/osmosis-mcp-4B-Q4_K_S.gguf"

if not MODEL_PATH.exists():
    raise Exception(f"MODEL_PATH is inavlid: {MODEL_PATH}")


# ============================= Vector Stores ==============================
# from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import InMemoryVectorStore
#
#
# def get_vector_store():
#     return InMemoryVectorStore()


# ============================= Document Loaders =============================
# from langchain.document_loaders.obsidian import ObsidianLoader


# ============================= Execution =============================

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=32,
    n_batch=512,
    n_threads=6,
    temperature=0.7,
    max_tokens=512,
    verbose=True,
)

prompt = PromptTemplate(
    template="Ответь на вопрос: {question}",
    input_variables=["question"],
)

chain = LLMChain(prompt=prompt, llm=llm)

if __name__ == "__main__":
    response = chain.invoke({"question": "Explain the concept of a vector store."})
    print(response)
