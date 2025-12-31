set dotenv-load
set export

OPENAI_URL := "http://127.0.0.1:1234/v1/"
LLM_MODEL := "mistralai/mistral-nemo-instruct-2407"
EMBEDDING_MODEL := "text-embedding-nomic-embed-text-v1.5"
EMBEDDING_MODEL_DIMENSIONS := "768"



[doc("")]
run filename *ARGS:
  uv run {{ filename }} {{ ARGS }}

[doc("""ex:
  just test pipelines/playground.py -m target 
  just test pipelines/playground.py -m target --verbose --memray --memray-bin-path=data
""")]
test target *ARGS:
  uv run pytest --disable-warnings --no-header --capture=no {{ ARGS }} {{ target }}

[doc("""ex:
  just memray flamegraph data/<filename.bin>
""")]
memray profile filename *ARGS:
  uv run memray {{ profile }} {{ filename }} {{ ARGS }}

lint:
  uv run mypy .

infra_up:
  podman-compose -f compose/qdrant.compose.yaml up -d
