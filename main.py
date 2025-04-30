import json
import os
from collections import OrderedDict

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_session import Session

from src import processing

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
# Configure Flask to use the filesystem for session storage
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = (
    "/tmp/flask_session"  # Directory to store session files
)
Session(app)

CORS(app)


@app.route("/kg_query", methods=["GET"])
@cross_origin()
def get_kg_data():
    try:

        # Create an empty array to store the values of "q"
        root_q_array = []

        # Get the value(s) of the "q" parameter from the URL
        root_q_values = request.args.getlist("q")

        # Append each value of "q" to the array
        for value in root_q_values:
            root_q_array.append(value)

        # Find the most similar file
        kg_content = processing.find_kg(root_q_array)
        # Create a response dictionary with the value of "q"
        response = {"kg_data": kg_content}

        # Return the response as JSON
        return jsonify(response)
    except Exception as e:
        print(e)


@app.route("/llm", methods=["POST"])
@cross_origin()
def send_prompt_llm():
    try:
        user_query = request.get_json()["query"]
        query_type = request.get_json().get("query_type")  # optional
        response = {}

        if query_type == "full":

            # user is requering ... get all relevant answers
            entities_dict = processing.get_knowledge_graph(user_query)
            query_idea_list = processing.generate_query_ideas(user_query)
            excerpts_dict = processing.run_semantic_search(user_query)
            excerpts_dict_synthesis = processing.remove_thumbnails(excerpts_dict)
            answer = processing.get_synthesis(user_query, excerpts_dict_synthesis)

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
            return jsonify(response)

    except Exception as e:
        print(e)
        # Return error response
        return jsonify(
            {
                "status": "failed",
                "message": "an error occurred",
                "session_id": None,
                "answers": [],
                "entities": [],
                "prompts": [],
            }
        )


if __name__ == "__main__":
    app.run()
