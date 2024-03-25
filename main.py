import imp
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
 
import websockets
# web
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS, cross_origin

# import custom utils functions 
import utils.processing_modules as processing_modules

# load enviroment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

CORS(app)

print(openai.VERSION)
 
# OpenAI API configuration
openai.api_type = "azure"
openai.api_key = os.getenv("api_key_azure")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("api_version")
openai_deployment = "sdgi-gpt-35-turbo-16k"
client = AzureOpenAI(
  api_key = os.getenv("api_key_azure"),  
  api_version = os.getenv("api_version"),
  azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
)
embedding_model = os.getenv("USER_QUERY_EMBEDDING_ENGINE") 

# Function to check API key
def require_api_key(api_key):
    valid_keys = [os.getenv('API_ACCESS_KEY') ]  # Replace with your valid API keys
    return api_key in valid_keys

# Apply the before_request decorator to all routes
@app.before_request
def check_api_key():
    # if request.endpoint != 'static_access':
    api_key = request.headers.get('API-Key')
    if api_key != os.getenv('API_ACCESS_KEY'):
        return jsonify({'error': 'Unauthorized access'}), 401


@app.route('/llm', methods = ['POST'])
@cross_origin() 
def send_promt_llm():
    try: 
        user_query = request.get_json()['query']

         ##run processing modules (in parallel)
        entities_dict=processing_modules.knowledgeGraphModule(user_query, openai_deployment)
        excerpts_dict=processing_modules.semanticSearchModule(user_query,client,embedding_model)
        indicators_dict=processing_modules.indicatorsModule(user_query) ##lower priority
        query_idea_list=processing_modules.queryIdeationModule(user_query, openai_deployment) ##lower priority
         
        ##synthesis module
        answer= processing_modules.synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict, openai_deployment)
     
        # return jsonify({
        #         "user_query":user_query,
        #         "answer":answer,
        #         "sources":'',
        #         "query_ideas":'',
        #         "entities":list(entities_dict["entities"].keys())       
        #     })

        # ##structure response
        return jsonify({
                "user_query":user_query,
                "answer":answer,
                "sources":excerpts_dict,
                "query_ideas":query_idea_list,
                "entities":list(entities_dict["entities"].keys())       
            })
    except Exception as e:
        print(e)
        # I did not find anything from the existing documents
        return jsonify(
                     {
                        'status':'failed',
                        'message': 'an error occured',
                        'answers': [], 
                        'entities': [],
                        'prompts': []
                         })



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
            await websocket.send(json.dumps({"indicators_dict": indicators_dict}))

            query_idea_list = processing_modules.queryIdeationModule(user_query, openai_deployment)
            await websocket.send(json.dumps({"query_idea_list": query_idea_list}))
            
            excerpts_dict = processing_modules.semanticSearchModule(user_query, client, embedding_model)
            await websocket.send(json.dumps({"excerpts_dict": excerpts_dict}))
            
            answer = processing_modules.synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict,
                                                        openai_deployment)
            await websocket.send(json.dumps({"answer": answer}))

        except Exception as e:
            error_response = {"error": str(e)}
            await websocket.send(json.dumps(error_response))


start_server = websockets.serve(handler, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
 
asyncio.get_event_loop().run_forever()

# if __name__ == "__main__":
#     app.run()
 
 