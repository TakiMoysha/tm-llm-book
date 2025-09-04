set dotenv-load

[doc("")]
run filename *ARGS:
  uv run {{ filename }} {{ ARGS }}

[doc("")]
test target *ARGS:
    uv run pytest --verbose --capture=no {{ ARGS }} {{ target }}
