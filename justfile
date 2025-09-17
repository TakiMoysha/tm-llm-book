set dotenv-load

[doc("")]
run filename *ARGS:
  uv run {{ filename }} {{ ARGS }}

[doc("ex: just test scripts/playground.py -m target")]
test target *ARGS:
    uv run pytest --disable-warnings --verbose --capture=no {{ ARGS }} {{ target }}
