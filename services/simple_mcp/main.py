import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
mcp = FastMCP("simple crawler")


def run_cli():
    logging.info("Starting MCP server with config: ...")

    try:
        mcp.run()
    except Exception as err:
        logging.error(f"Failed to run MCP server: {err}")


if __name__ == "__main__":
    run_cli()
