"""
Entry point to the API.
"""

import json
from collections import OrderedDict
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel

from src import processing

load_dotenv()


app = FastAPI()


class Message(BaseModel):
    content: str
    full: bool = False


@app.get(path="/kg_query")
async def get_kg_data(q: Annotated[list[str], Query()]):
    # Find the most similar file
    kg_content = processing.find_kg(q)
    # Create a response dictionary with the value of "q"
    response = {"kg_data": kg_content}

    # Return the response as JSON
    return response


@app.post(path="/llm")
async def send_prompt_llm(message: Message):
    try:
        user_query = message.content
        response = {}

        if message.full:

            # user is requering ... get all relevant answers
            entities_dict = processing.get_knowledge_graph(user_query)
            query_idea_list = processing.generate_query_ideas(user_query)
            excerpts_dict = processing.process_queries(user_query)
            excerpts_dict_synthesis = processing.remove_thumbnails(excerpts_dict)
            answer = processing.get_answer(user_query, excerpts_dict_synthesis)

            response = OrderedDict(
                [
                    ("answer", answer),
                    ("user_query", user_query),
                    (
                        "entities",
                        (list(entities_dict["entities"]) if entities_dict else []),
                    ),
                    ("query_ideas", query_idea_list if query_idea_list else []),
                    ("excerpts_dict", excerpts_dict),
                    ("indicators_dict", {}),
                ]
            )

            # Convert the response to a JSON string and then back to a dictionary to preserve order
            response_json = json.dumps(response, indent=4)

            # Return the response
            return response_json
        else:

            entities_dict = processing.get_knowledge_graph(user_query)
            query_idea_list = processing.generate_query_ideas(user_query)

            # Get results from completed futures
            entities_array = list(entities_dict["entities"]) if entities_dict else []

            kg_content = processing.find_kg(entities_array)
            response = {
                "answer": "Processing final answer... ",
                "user_query": user_query,
                "entities": entities_array,
                "query_ideas": query_idea_list if query_idea_list else [],
                "excerpts_dict": {},
                "indicators_dict": {},
                "kg_data": kg_content,
            }

            # Return the response
            return response

    except Exception as e:
        print(e)
        # Return error response
        return {
            "status": "failed",
            "message": "an error occurred",
            "session_id": None,
            "answers": [],
            "entities": [],
            "prompts": [],
        }
