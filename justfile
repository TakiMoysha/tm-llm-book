set dotenv-load
set export

OPENAI_URL := "http://127.0.0.1:1234/v1/"
LLM_MODEL := "mistralai/mistral-nemo-instruct-2407"
EMBEDDING_MODEL := "text-embedding-nomic-embed-text-v1.5"
EMBEDDING_MODEL_DIMENSIONS := "1024"



[doc("")]
run filename *ARGS:
  uv run {{ filename }} {{ ARGS }}

[doc("ex: just test scripts/playground.py -m target")]
test target *ARGS:
    uv run pytest --disable-warnings --verbose --capture=no {{ ARGS }} {{ target }}
