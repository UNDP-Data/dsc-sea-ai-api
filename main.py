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

# web
from flask import Flask, request, jsonify
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

if __name__ == "__main__":
    app.run()
    
