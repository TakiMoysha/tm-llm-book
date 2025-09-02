
# ex: just init_service simple_mcp
init_service service_name:
  uv init --project=services/{{ service_name }}

[doc("""added new dependency to service_name, ex: just add langchain_rag langchain""")]
service_add service_name new_dependency:
  uv --directory=services/{{ service_name }} add --project=services/{{ service_name }} {{ new_dependency }}

# ex: just 
[doc("""run service, ex: just service_run langchain_rag""")]
service_run service_name *ARGS:
  uv --directory=services/{{ service_name }} run --project=services/{{ service_name }} {{ ARGS }}

# ex: just prompt_server.py
research_mcp file_name *ARGS:
  uv --directory=services/research_mcp run {{ file_name }} {{ ARGS }}

# ex: just simple_rag
simple_rag file_name *ARGS:
  uv --directory=services/simple_rag run {{ file_name }} {{ ARGS }}


