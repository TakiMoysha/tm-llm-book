set dotenv-load

[doc("")]
run filename ARGS:
  uv run {{ filename }} {{ ARGS }}
