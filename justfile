
# ex: just init_service simple_mcp
init_service service_name:
  uv init --project=services/{{ service_name }}

# ex: just 
service_add service_name dependency *ARGS:
  uv add --project=services/{{ service_name }} {{ dependency }} {{ ARGS }}

# ex: just 
service_run service_name *ARGS:
  uv run --project=services/{{ service_name }} {{ ARGS }}

# ex: just prompt_server.py
research_mcp file_name *ARGS:
  uv --directory=services/research_mcp run {{ file_name }} {{ ARGS }}
