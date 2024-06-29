from concurrent.futures import process
import imp
from pickle import TRUE
from urllib import response
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
import re
import uuid  # for generating unique session IDs
import utils.openai_call as openai_call
from collections import OrderedDict

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



@app.route('/kg_query', methods=['GET'])
@cross_origin()
def get_kg_data():
    try: 

        # Create an empty array to store the values of "q"
        root_q_array = []
        
        # Get the value(s) of the "q" parameter from the URL
        root_q_values = request.args.getlist('q')
        
        # Append each value of "q" to the array
        for value in root_q_values:
            root_q_array.append(value)
        
        data_dir = "data/KG"
        # Find the most similar file
        kg_content = processing_modules.find_kg(root_q_array, data_dir)
        # Create a response dictionary with the value of "q"
        response = {
            "kg_data": kg_content
        }
        
        # Return the response as JSON
        return jsonify(response)
    except Exception as e:
        print(e)


@app.route('/llm', methods = ['POST'])
@cross_origin() 
def send_prompt_llm():
    try: 
        user_query = request.get_json()['query']
        # session_id = str(uuid.uuid4())
        # session_id_query = request.get_json()['session_id'] #optional
        query_type = request.get_json().get('query_type') #optional
        print(user_query)
        response = {}
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
                    prompt_formattings = ""
                    # Get results from completed futures
                    entities_dict = future_entities.result()

                    indicators_dict = future_indicators.result()
                    query_idea_list = future_query_ideas.result()

                    isInitialRun = False
                    future_excerpts = executor.submit(run_module, processing_modules.semanticSearchModule, client, embedding_model,isInitialRun)
                    excerpts_dict = future_excerpts.result()

                    excerpts_dict_synthesis = processing_modules.remove_thumbnails(future_excerpts.result())

                    # Run synthesis module
                    answer = processing_modules.synthesisModule(user_query, entities_dict, excerpts_dict_synthesis, indicators_dict, openai_deployment, prompt_formattings)
                    pattern =  re.compile(r'[^.]*\.')  #re.compile(r'<li>(.*?)</li>')
                    # Find all matches
                    content_array = pattern.findall(answer)
                    sources = excerpts_dict


                    results = []
                    # print(content_array)
                    limiter = 0
                    
                    for element in content_array:
                        for doc_id, doc_info in sources.items():
                            title_similarity = processing_modules.calculate_context_similarity(element, doc_info['document_title']) or 0 
                            extract_similarity = processing_modules.calculate_context_similarity(element, doc_info['extract']) or 0
                            # summary_similarity = calculate_context_similarity(element, doc_info['summary'])
                            
                            if title_similarity > 0.7 and extract_similarity > 0.8 and limiter < 10:
                                result = {
                                            'element': element,
                                            'title': doc_info['document_title'],
                                            'extract': doc_info['extract'],
                                            'extract': doc_info['extract'],
                                            'link': doc_info['document_title'],
                                            'doc_id': doc_id,
                                            'title_similarity': float(title_similarity),
                                            'extract_similarity': float(extract_similarity)
                                            # 'summary_similarity': float(summary_similarity)
                                        }
                                results.append(result)
                                limiter += 1

                    for result in results:
                        citation_fixes = openai_call.callOpenAI(f"Given the below: {result} Create an output that mixes Element, Document extract and Summary into one output while still maintaining the context of the Element. Your final output answer length should not be more than 200 words. Also avoid using links, sources and references. ", openai_deployment)
                        result['citation_fixes'] = citation_fixes
                        result

            
                    content = answer
                    counter = 0
                    # Loop through each JSON object and replace the element with citation_fixes in the content
                    for result in results:
                        counter += 1

                        content = content.replace(result['element'], f""" {result['citation_fixes']} <a href='{result['link']}' data-id='{result['doc_id']}'>[{counter}]</a> <br/>\n\n""")
                        
                    sorted_sources = sources
                    print(sorted_sources)
                        #Send initial response to user while processing final answer on final documents
                    # response = {
                    #         "answer": content.replace("\n","<br/>"),
                    #         "user_query": user_query,
                    #         "entities": list(entities_dict["entities"].keys()) if entities_dict else [],
                    #         "query_ideas": query_idea_list if query_idea_list else [],
                    #         "excerpts_dict" : sorted_sources,
                    #         "indicators_dict": indicators_dict
                    #     }

                    #final cleanup using openAI
                    cleanup_content = openai_call.callOpenAI(f""" Ignore previous commands !!!
                                                    Strictly follow the below:
                                                    Give the sentence. I want to to fix the citation formatings only. Don't add any answer to it.
                                                    1. make sure  links are all in a citation format [n] where n represent an integer and must link to the document e.g content<a href='url-here'>[1]</a>  !!!!
                                                    2.  The citations must be numbered in an ordered manner. Fix and return the output. !!!
                                                    3. remove all foot notes or references. !!! 
                                                    4. The citations MUST BE LINK to the docs e.g <a href='url-here'>[1]</a>  never use without LINKS !!!
                                                    5. Output should retains HTML formattings. Never adjust a ciation without it being an anchor link. !!!
                                                    6. Remeber only format the answer citations. Don't add or remove any. !!!
                                                    7. Don't generate any link or so. Just use the answer as it is and adjust the citations as instructed above
                                                    SENTENCE: {content}  
                                                """, openai_deployment)
                    cleanup_content = cleanup_content.replace("\n","")
                    cleanup_content = processing_modules.cleanCitation(cleanup_content)
                    cleanup_content = processing_modules.check_links_and_process_html(cleanup_content, sorted_sources)
                    # Construct the final response using OrderedDict to preserve key order
                    response = OrderedDict([
                        ("answer", cleanup_content.replace("\n", "")),
                        ("user_query", user_query),
                        ("entities", list(entities_dict["entities"].keys()) if entities_dict else []),
                        ("query_ideas", query_idea_list if query_idea_list else []),
                        ("excerpts_dict", sorted_sources),
                        ("indicators_dict", indicators_dict)
                    ])

                    # Convert the response to a JSON string and then back to a dictionary to preserve order
                    response_json = json.dumps(response, indent=4)
                    # final_response = json.loads(response_json)

                
                # Return the response
                return  response_json
        else : 
            
            # Create a thread pool executor - run all in parallel to reduce ttl
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit processing modules to the executor
                future_entities = executor.submit(run_module, processing_modules.knowledgeGraphModule, openai_deployment)
                future_indicators = {} #executor.submit(run_module, processing_modules.indicatorsModule)
                future_query_ideas = executor.submit(run_module, processing_modules.queryIdeationModule, openai_deployment)

                # Get results from completed futures
                entities_dict = future_entities.result()
                indicators_dict = {} #future_indicators.result()
                query_idea_list = future_query_ideas.result()
                isInitialRun = TRUE
                excerpts_dict = {}
                entities_array = list(entities_dict["entities"].keys()) if entities_dict else []
                data_dir = "data/KG"
                kg_content = processing_modules.find_kg(entities_array, data_dir)
                response = {
                    "answer": "Processing final answer... ",
                    "user_query": user_query,
                    "entities": entities_array,
                    "query_ideas": query_idea_list if query_idea_list else [],
                    "excerpts_dict" : excerpts_dict,
                    "indicators_dict": indicators_dict,
                    "kg_data": kg_content
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
