import imp
from pickle import TRUE
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import ast
from openai import AzureOpenAI
import faiss
import numpy as np
import json
import asyncio
import concurrent.futures

import uuid  # for generating unique session IDs

import websockets
# web
from flask import Flask, request, jsonify,session
from flask_cors import CORS, cross_origin
from flask_session import Session

# import custom utils functions 
import utils.processing_modules as processing_modules

# load enviroment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
# Configure Flask to use the filesystem for session storage
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'  # Directory to store session files
Session(app)

CORS(app)
 

print(openai.VERSION)
 
# OpenAI API configuration
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("api_version")
openai_deployment = "sdgi-gpt-35-turbo-16k"
client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = os.getenv("api_version"),
  azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
)
embedding_model = os.getenv("USER_QUERY_EMBEDDING_ENGINE") 

# Function to check API key
def require_api_key(api_key):
    valid_keys = [os.getenv('API_ACCESS_KEY') ]  # Replace with your valid API keys
    return api_key in valid_keys

# Apply the before_request decorator to all routes
# @app.before_request
# def check_api_key():
#     # if request.endpoint != 'static_access':
#     api_key = request.headers.get('API-Key')
#     if api_key != os.getenv('API_ACCESS_KEY'):
#         return jsonify({'error': 'Unauthorized access'}), 401


@app.route('/llm', methods = ['POST'])
@cross_origin() 
def send_prompt_llm():
    try: 
        user_query = request.get_json()['query']
        # session_id = str(uuid.uuid4())
        # session_id_query = request.get_json()['session_id'] #optional
        query_type = request.get_json().get('query_type') #optional
        print(user_query)
        
        # print(get_session)
        # Define a function to run each processing module
        def run_module(module_func, *args):
            print('running')
            return module_func(user_query, *args)

        if query_type == 'full':
                print("in session")
                print(query_type)
                # Delete the session
                # session.clear()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                #user is requering ... get all relevant answers
                    future_entities = executor.submit(run_module, processing_modules.knowledgeGraphModule, openai_deployment)
                    future_indicators = executor.submit(run_module, processing_modules.indicatorsModule)
                    future_query_ideas = executor.submit(run_module, processing_modules.queryIdeationModule, openai_deployment)

                    # Get results from completed futures
                    entities_dict = future_entities.result()

                    indicators_dict = future_indicators.result()
                    query_idea_list = future_query_ideas.result()

                    isInitialRun = False
                    future_excerpts = executor.submit(run_module, processing_modules.semanticSearchModule, client, embedding_model,isInitialRun)
                    excerpts_dict = future_excerpts.result()

                    # Run synthesis module
                    answer = processing_modules.synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict, openai_deployment)
                    
                    #Send initial response to user while processing final answer on final documents
                    response = {
                        "answer": answer,
                        "user_query": user_query,
                        "entities": list(entities_dict["entities"].keys()) if entities_dict else [],
                        "query_ideas": query_idea_list if query_idea_list else [],
                        "excerpts_dict" : excerpts_dict,
                        "indicators_dict": indicators_dict
                    }
                    
                            # Return the response
                return jsonify(response)
        else : 
            
            # Create a thread pool executor - run all in parallel to reduce ttl
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit processing modules to the executor
                future_entities = executor.submit(run_module, processing_modules.knowledgeGraphModule, openai_deployment)
                future_indicators =executor.submit(run_module, processing_modules.indicatorsModule)
                future_query_ideas = executor.submit(run_module, processing_modules.queryIdeationModule, openai_deployment)

                # Get results from completed futures
                entities_dict = future_entities.result()
                indicators_dict = future_indicators.result()
                query_idea_list = future_query_ideas.result()

                isInitialRun = TRUE
                future_excerpts = executor.submit(run_module, processing_modules.semanticSearchModule, client, embedding_model,isInitialRun)
                excerpts_dict = future_excerpts.result()

                # Run synthesis module
                # answer = processing_modules.synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict, openai_deployment)
                
                #Send initial response to user while processing final answer on final documents
                response = {
                    "answer": "Processing final answer... re-query to retrieve final answer and documents using session id",
                    "user_query": user_query,
                    "entities": list(entities_dict["entities"].keys()) if entities_dict else [],
                    "query_ideas": query_idea_list if query_idea_list else [],
                    "excerpts_dict" : excerpts_dict,
                    "indicators_dict": indicators_dict
                }

                # session['session_id'] = session_id #save the session id



            # Return the response
            return jsonify(response)

    except Exception as e:
        print(e)
        # Return error response
        return jsonify(
            {
                'status': 'failed',
                'message': 'an error occurred',
                'session_id': None,
                'answers': [], 
                'entities': [],
                'prompts': []
            }
        )


# create handler for each connection 
async def handler(websocket, path):
    async for message in websocket:
        try:
            data = json.loads(message)
            user_query = data.get('query')

            # Process the user query step by step and send responses
            entities_dict = processing_modules.knowledgeGraphModule(user_query, openai_deployment)
            await websocket.send(json.dumps({"entities_dict": entities_dict}))
            
            indicators_dict = processing_modules.indicatorsModule(user_query)
            # await websocket.send(json.dumps({"indicators_dict": indicators_dict}))

            query_idea_list = processing_modules.queryIdeationModule(user_query, openai_deployment)
            await websocket.send(json.dumps({"query_idea_list": query_idea_list}))
            
            excerpts_dict = processing_modules.semanticSearchModule(user_query, client, embedding_model, isInitialRun = TRUE)
            await websocket.send(json.dumps({"excerpts_dict": excerpts_dict}))
            
            answer = processing_modules.synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict,
                                                        openai_deployment)
            await websocket.send(json.dumps({"answer": answer}))

        except Exception as e:
            error_response = {"error": str(e)}
            await websocket.send(json.dumps(error_response))


# start_server = websockets.serve(handler, "", 5000)
# start_server = websockets.serve(handler, None, 0)

# start_server = websockets.serve(handler, "0.0.0.0", port=5000)
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()

if __name__ == "__main__": 
    app.run()
