import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


MCP_JSON = {
    "inputs": [],
    "servers": {"microsoft_mcp_learning": {"command": "uv", "args": ["--directory", ".", "run", "prompt_server.py"]}},
}
mcp = FastMCP("prompt demo server")

TLevels = Literal["begginer", "intermediate", "advanced"]


# ==========================================================================================
# PROMPTS - pre-written templtes that help users accomplish specific stasks


@mcp.prompt()
def python_topics(level: TLevels = "begginer") -> str:
    learning_levels = {
        "begginer": "for someone who is learning python",
        "intermediate": "for someone with some python experience",
        "advanced": "for someone with python experience",
    }

    if level not in learning_levels.keys():
        raise ValueError(f"Invalid level: {level}")

    prompt = f"generate 5 python topics {learning_levels[level]}"
    return prompt


# ==========================================================================================
# TOOLS - tools that help users accomplish specific tasks


@dataclass
class Exercise:
    title: str
    description: str
    hint: str
    solution: str
    story_point: int


exercises_db = {
    "beginner": [
        Exercise(
            title="Hello World",
            description="Write a program that prints 'Hello, Python!' to the console",
            hint="Try using print()",
            solution="print('Hello, World!')",
            story_point=1,
        )
    ],
    "intermediate": [
        Exercise(
            title="List Comprehension",
            description="Create a list of squares for numbers 1-10 using list comprehension",
            hint="Use the syntax: [expression for item in range()]",
            solution="squares = [x**2 for x in range(1, 11)]\nprint(squares)",
            story_point=3,
        )
    ],
    "advanced": [
        Exercise(
            title="Decorator Pattern",
            description="Create a decorator that times how long a function takes to run",
            hint="Use time.time() before and after the function call",
            solution="import time\nfrom functools import wraps\n\ndef timer(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end-start:.4f} seconds')\n        return result\n    return wrapper",
            story_point=5,
        )
    ],
}


@mcp.prompt()
async def generate_exersice(topic: str, level: str = "begginer") -> str:
    return f"Generate Python exersice on topic: {topic} for level: {level}"


@mcp.tool()
async def generrate_and_create_excersices(
    topic: str,
    level: str = "begginer",
    ctx: Context | None = None,
) -> str:
    try:
        prompt_text = await generate_exersice(topic, level)
        response = await ctx.session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt_text))], max_tokens=2000
        )
        response_text = response.content.text if response.content else ""
        exercises_data = json.loads(response_text)

        exercises_db[level] = []
        for ex in exercises_data[level]:
            exercises_db[level].append(
                Exercise(
                    title=ex["title"],
                    description=ex["desc"],
                    hint=ex["hint"],
                    solution=ex["solution"],
                    story_point=ex["story_point"],
                )
            )

        return f"Successfully generated {len(exercises_data[level])} exersices for topic: {topic} and level: {level}"
    except json.JSONDecodeError as err:
        return f"JSONDecodeError: {err}"
    except Exception as err:
        return f"Failed to generate exersices for topic: {topic} and level: {level}: {err}"


@mcp.tool()
async def list_exercises() -> str:
    if not exercises_db:
        return "No exercises found, use `generrate_and_create_excersices` first"

    result = []
    for level, exercises in exercises_db.items():
        result.append(f"Level: {level.upper()}")
        for i, exercise in enumerate(exercises, 1):
            result.append(f"Title: {i + 1}.{exercise.title}")
            result.append(f"Description: {exercise.description}")
            result.append(f"Hint: {exercise.hint}")
            result.append(f"Solution: {exercise.solution}")
            result.append(f"Story point: {exercise.story_point}/5")
            result.append("")

    return "\n".join(result)


# ==========================================================================================
# RESOUCES - tools that help users accomplish specific tasks

study_progress_file = os.path.join(os.path.dirname(__file__), "study_progress.json")
beginner_exercises_file = os.path.join(os.path.dirname(__file__), "beginner_exercises.json")


@mcp.resource("user://study-progress/{username}")
async def get_study_progress(username: str) -> str:
    try:
        with open(study_progress_file, "r") as f:
            study_progress = json.load(f)

        if study_progress.get("username") == username:
            return json.dumps(study_progress, indent=2)
        else:
            return json.dumps({"error": f"Study progress not found for user: {username}"})

    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {study_progress_file}"})
    except json.JSONDecodeError as err:
        return json.dumps({"error": f"JSONDecodeError: {err}"})
    except Exception as err:
        return json.dumps({"error": f"Failed to get study progress for user: {username}: {err}"})


@mcp.resource("user://exercises/{level}")
async def list_exercises_for_level(level: str) -> str:
    try:
        if level != "begginer":
            return json.dumps({"error": f"no exercises for level: {level}"})

        with open(beginner_exercises_file, "r") as f:
            exercises = json.load(f)

        return json.dumps(exercises, indent=2)

    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {beginner_exercises_file}"})
    except json.JSONDecodeError as err:
        return json.dumps({"error": f"JSONDecodeError: {err}"})


@mcp.tool()
async def get_users_progress(username: str, ctx: Context | None = None) -> str:
    try:
        user_progress_json = await get_study_progress(username)
        user_progress = json.loads(user_progress_json)
        prompt_text = f"""Here is the study progress for user: {username}\n{json.dumps(user_progress, indent=2)}.
        Return it to the user and suggest some exercises for them to practice"""

        response = await ctx.session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt_text))],
            max_tokens=2000,
        )
        response_text = response.content.text if response.content else ""
        return response_text
    except json.JSONDecodeError as err:
        return f"JSONDecodeError: {err}"
    except Exception as err:
        return f"Failed to get study progress for user: {username}: {err}"


# ==========================================================================================


def run_cli():
    logging.info("Starting MCP server with config: ...")

    try:
        mcp.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_cli()
