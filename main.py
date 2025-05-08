"""
Entry point to the API.
"""

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
    user_query = message.content
    response = {}

    # user is requering ... get all relevant answers
    entities_dict = processing.get_knowledge_graph(user_query)
    query_idea_list = processing.generate_query_ideas(user_query)
    entities_array = list(entities_dict["entities"]) if entities_dict else []
    if message.full:
        excerpts_dict = processing.process_queries(user_query)
        excerpts_dict_synthesis = processing.remove_thumbnails(excerpts_dict)
        answer = processing.get_answer(user_query, excerpts_dict_synthesis)
        kg_content = None
    else:
        excerpts_dict, answer = {}, "Processing final answer... "
        kg_content = processing.find_kg(entities_array)

    response = OrderedDict(
        [
            ("answer", answer),
            ("user_query", user_query),
            ("entities", entities_array),
            ("query_ideas", query_idea_list if query_idea_list else []),
            ("excerpts_dict", excerpts_dict),
            ("indicators_dict", {}),
            ("kg_data", kg_content),
        ]
    )

    # Return the response
    return response
