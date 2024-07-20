
from html import entities
import utils.openai_call as openai_call
import openai
import ast
import pandas as pd
import faiss
import numpy as np
import pycountry
import re
import copy

import awoc
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm")
import os
import concurrent.futures
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import base64
import json
import copy
from country_named_entity_recognition import find_countries

# import custom utils functions 
import utils.processing_modules as processing_modules
# import custom utils functions 
import utils.indicator as indicator_module
from bs4 import BeautifulSoup

from awoc import AWOC

# model = transformers.BertModel.from_pretrained('bert-base-uncased')
# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_pickle('./models/df_embed_EN_All_V4.pkl')

# Extract entities for the query and return the extract entities as an array
def extractEntitiesFromQuery(user_query, openai_deployment):
    prompt = f"""
    Extract entities from the following user query: \"{user_query}\" and return output in array format.
    
    -Entities should be directly related to the domain or topic of interest. They should represent important concepts that contribute to the understanding of the subject matter.
    -Each entity in the knowledge graph should be distinct and have a unique identifier. This ensures clarity and avoids ambiguity when establishing relationships between entities.
    -You Must return output in array format e.g  ['entity1','entity2'] !!!
    -Avoid adding new lines or breaking spaces to your output. Array should be single dimension and single line !!!
 
    """
    entity_list = openai_call.callOpenAI(prompt, openai_deployment)   
    return entity_list




def generate_thumbnail_from_pdf(pdf_url, page_number=0, thumbnail_size=(100, 100)):
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_url)
        
        # Get the specified page
        page = pdf_document.load_page(page_number)
        
        # Get the image bytes of the page thumbnail
        image_bytes = page.get_pixmap(matrix=fitz.Matrix(1, 1)).tobytes()

        # Create PIL image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Resize the image to thumbnail size
        image.thumbnail(thumbnail_size)
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode()

        return thumbnail_base64

    except Exception as e:
        print("Error:", e)
        return None


## module to get information on the entities from user query using the KG
def knowledgeGraphModule(user_query, openai_deployment):
    
    # generate list of entities based on user query
    entity_list = extractEntitiesFromQuery(user_query, openai_deployment)
    my_list = ast.literal_eval(entity_list)
    prompt_summarise_entites = f"""
    Summarize all relations between all the entities : {my_list}
    """
    summarise_entities = openai_call.callOpenAI(prompt_summarise_entites, openai_deployment)
    # Initialize an empty dictionary to store information
    entities_dict = {
        "relations": summarise_entities,
        "entities": {}
    }
    # Loop through each entity in the list
    for entity in my_list:
        # Fetch information about the entity from your knowledge graph
        prompt = f"Give me a short description 50 words of {entity}"
        entity_info = '' #openai_call.callOpenAI(prompt, openai_deployment)
        # Add the entity information to the dictionary
        entities_dict["entities"][entity] = entity_info
    
    return entities_dict

def find_mentioned_countries(text):
    countries = set()
    
    # Tokenize the text using regular expressions to preserve punctuation marks
    words = re.findall(r'\w+|[^\w\s]', text)
    text = ' '.join(words)  # Join the tokens back into a string
    
    # Get a list of all country names
    all_countries = {country.name: country for country in pycountry.countries}
    
    # Check for multi-word country names first to avoid partial matches
    for name in sorted(all_countries.keys(), key=lambda x: len(x), reverse=True):
        if name in text:
            countries.add(all_countries[name].name)
            text = text.replace(name, '')  # Remove the found country name from the text to avoid duplicates

    return list(countries)



# def find_mentioned_countries(text):
#     countries = set()
    
#     # Tokenize the text using regular expressions to preserve punctuation marks
#     words = re.findall(r'\w+|[^\w\s]', text)
#     text = ' '.join(words)  # Join the tokens back into a string
    
#     for word in text.split():
#         try:
#             country = pycountry.countries.get(name=word) #pycountry.countries.lookup(word)
#             if country != None : 
#                countries.add(country.name)
#         except LookupError:
#             pass
    
#     return list(countries)


'''
Previous 'find_mentioned_countries' can detect countries when they are formed correctly.

'''
# Extract mentioned countries' ISO3 code
def find_mentioned_country_code(user_query):
    countries = set()
    extracted_countries = find_mentioned_countries(user_query)
    
    for country in extracted_countries:
        try:
            country_info = pycountry.countries.get(name=country)
            if country_info:
                countries.add(country_info.alpha_3)
        except LookupError:
            pass
    
    # If no countries are found, check for continent mentions
    if not countries:
        words = re.findall(r'\w+|[^\w\s]', user_query)
        text = ' '.join(words)  # Join the tokens back into a string
    
        world_info = AWOC()
        all_continents = set([continent.lower() for continent in world_info.get_continents_list()])
        for word in text.split():
            word = word.lower()
            if word in all_continents:
                target_countries = world_info.get_countries_list_of(word)
                
                for country in target_countries:
                    countries.add(world_info.get_country_data(country)['ISO3'])
    
    return countries
def filter_country(user_query):
    mentioned_countries = find_mentioned_country_code(user_query)
    print(mentioned_countries)
    # Check if mentioned_countries is not empty
    if mentioned_countries:
        country = mentioned_countries[0]
        return df[df['Country Name'] == country]
    else:
        # Handle the case where no countries were mentioned
        return None  # Or return an empty DataFrame or any other suitable value

 

# Function to calculate Jaccard similarity between two texts
def jaccard_similarity(text1, text2):
    # Tokenize texts
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0

# Load the English language model
# Function to calculate the average word embedding for a sentence
def average_word_embedding_old(sentence):
    # Parse the sentence using SpaCy
    doc = nlp(sentence)
    # Get word vectors and average them
    word_vectors = [token.vector for token in doc if token.has_vector]
    if not word_vectors:
        return None
    return np.mean(word_vectors, axis=0)

def average_word_embedding(sentence):
    if sentence is None:
        sentence = ""
    
    # Parse the sentence using SpaCy
    doc = nlp(sentence)
    
    # Get word vectors and average them
    vectors = [token.vector for token in doc if token.has_vector]
    if not vectors:
        return None
    
    avg_vector = sum(vectors) / len(vectors)
    return avg_vector

# Function to calculate context similarity between two sentences using word embedding averaging
def calculate_context_similarity(sentence1, sentence2):
    # Get average word embeddings for each sentence

    avg_embedding1 = average_word_embedding(sentence1)
    avg_embedding2 = average_word_embedding(sentence2)
    # print(avg_embedding1)
    # print(avg_embedding2)
    if avg_embedding1 is None or avg_embedding2 is None:
        return None
    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity([avg_embedding1], [avg_embedding2])[0][0]
    return similarity


#Simple helps
def title_contains_entity(entity, title):
    # Convert both entity and title to lowercase for case-insensitive comparison
    entity_lower = entity.lower()
    title_lower = title.lower()

    # Check if the lowercase entity is contained within the lowercase title
    if entity_lower in title_lower:
        return 1
    else:
        return 0



# Function to convert country codes to country names
def convert_codes_to_names(codes):
    code_to_name = {country.alpha_3: country.name for country in pycountry.countries}
    return {code_to_name.get(code, code) for code in codes}



# Function to filter DataFrame based on country names
def filter_dataframe_by_country_names(df, filtered_country_cde):
    filtered_dfs = []
    country_names = convert_codes_to_names(filtered_country_cde)
    code_to_name = {country.alpha_3: country.name for country in pycountry.countries}
    
    for code in filtered_country_cde:
        country_name = code_to_name.get(code, None)
        if country_name:
            filtered_df = df[df['Country Code'] == code]
            filtered_df['Country Name'] = country_name
            filtered_dfs.append(filtered_df)
    
    if filtered_dfs:
        result_df = pd.concat(filtered_dfs, ignore_index=True)
    else:
        result_df = pd.DataFrame()  # Return empty DataFrame if no matches
    
    return result_df



def average_word_context_embed(sentence):
    # Ensure the input is a string
    if not isinstance(sentence, str):
        return None
    
    # If the sentence is empty, return None
    if not sentence:
        return None
    
    # Parse the sentence using SpaCy
    doc = nlp(sentence)
    
    # Get word vectors and average them
    vectors = [token.vector for token in doc if token.has_vector]
    if vectors:
        avg_vector = np.mean(vectors, axis=0)
        return avg_vector
    else:
        return None

def calculate_context_bool(uq, doc_, threshold):
    avg_emb1 = average_word_context_embed(uq)
    avg_emb2 = average_word_context_embed(doc_)
    if avg_emb1 is None or avg_emb2 is None:
        return False
    
    similarity = np.dot(avg_emb1, avg_emb2) / (np.linalg.norm(avg_emb1) * np.linalg.norm(avg_emb2))
    return similarity > threshold  # Assuming 0.75 is the threshold for context similarity


 
def filter_semantics(user_query, isInitialRun): 
    filtered_df_country = pd.DataFrame()
    filtered_df_title_context = pd.DataFrame()
    merged_df = pd.DataFrame()

    filtered_df_country_code = find_mentioned_country_code(user_query)
    filtered_df_country = filter_dataframe_by_country_names(df, filtered_df_country_code)

    
    filtered_df_title_context = df[df['Document Title'].notnull() & df['Document Title'].apply(lambda title: calculate_context_bool(user_query, title, 0.65))]
    filtered_df_summary_context = df[df['Summary'].notnull() & df['Summary'].apply(lambda summary: calculate_context_bool(user_query, summary, 0.7))]
    
    # Ensure both DataFrames have the same columns before concatenating
    if 'Country Name' not in filtered_df_title_context.columns:
        filtered_df_title_context['Country Name'] = np.nan
    if 'Country Name' not in filtered_df_summary_context.columns:
        filtered_df_summary_context['Country Name'] = np.nan
    
    # Merge the two filtered DataFrames
    merged_df = pd.concat([filtered_df_country, filtered_df_summary_context, filtered_df_title_context])
    return merged_df



 
 
 
def search_embeddings(user_query, client, embedding_model, isInitialRun):
    # df_filtered = filter_semantics(user_query) if filter_semantics(user_query) is not None else None
    filtered_result = filter_semantics(user_query, isInitialRun)
    # Check if the result is not None before assigning it to df_filtered
    df_filtered = filtered_result if filtered_result is not None else None

    if df_filtered is not None and not df_filtered.empty:  # Check if DataFrame is not None and not empty
        length = len(df_filtered.head())
        filtered_embeddings_arrays = np.array(list(df_filtered['Embedding']))
        index = faiss.IndexFlatIP(filtered_embeddings_arrays.shape[1]) 
        index.add(filtered_embeddings_arrays)
        
        user_query_embedding = client.embeddings.create( 
                input=user_query ,model= embedding_model
            ).data[0].embedding

        k = min(5, length)
        distances, indices = index.search(np.array([user_query_embedding]), k)
        return df_filtered, distances, indices
    else:
        return None, None, None
        
 

# get answer
def get_answer(user_question, relevant_docs,openai_deployment): 

    formattings_html = f""" 
        Ignore previous
        Strictly follow the follow steps:
        Your output answer shoud be  in HTML syntax with HTML tags.
        Use HTML tags like < ul>, < ol>, < li>,  < strong>, < p>
        Only consider the inner part of the < body> tag.
        ALWAYS use the following tag for new lines: < br />
        Do not add CSS attributes.
        Include links and citations at all!!!
        Your final answer must be formatted in HTML format !!!

        - Only provide links in citations. Never link outside citations or refer 
        Example 
        <a href="LINK">[n]</a> - correct
        <a href="LINK">text content</a> - wrong
        Where n is integer and LINK is a url
    """
    formattings = f""" 
        You can use relevant information in the docs to answer also: 

        DOCS: {relevant_docs}
        
       """
    messages = [
        {"role": "system", "content":f"""You are a helpful assistant and a professional researcher with many years of experience in answering questions. Give answer to the user's inquiry. {formattings_html}"""
        },
        {'role': 'user', 'content': f"""{formattings} 
                                        {user_question}
                                        
                                         {formattings_html}
                                        """},
    ]
        
    response_entities = openai.chat.completions.create(
                    model=openai_deployment,
                    temperature=0.3,
                    messages=messages,
                    top_p=0.8,
                    frequency_penalty=0.6,
                    presence_penalty=0.8

                )
    response = response_entities.choices[0].message.content
    # Define the regex pattern to match digits followed by '. do'
    pattern = r'\d+\. do'

    # Remove matches from the text
    cleaned_text = re.sub(pattern, '', response)

    # # Optionally, clean up any extra spaces or punctuation left behind
    # cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()


    return cleaned_text


def check_links_and_process_html(html, content_dict):
    soup = BeautifulSoup(html, 'html.parser')
    
    for a in soup.find_all('a'):
        ref_text = a.get_text()
        if ref_text.startswith('[') and ref_text.endswith(']'):
            href = a.get('href')
            if not any(d['document_link'] == href for d in content_dict.values()):
                a.decompose()
    
    result = str(soup)
    return result


def check_links_and_process_html____(html, content_dict):
    soup = BeautifulSoup(html, 'html.parser')
    link_count = 1

    for a in soup.find_all('a'):
        href = a.get('href')
        if any(d['link'] == href for d in content_dict.values()):
            # Ensure the link adheres to the format <a href="LINK HERE">[n]</a>
            a.string = f'{a.get_text()} [{link_count}]'
            link_count += 1
        else:
            a.decompose()

    result = str(soup)
    return result


def sort_by_relevancy(result_dict):
    # Convert the dictionary to a list of tuples (doc_id, info)
    result_list = list(result_dict.items())
    
    # Reverse the list
    result_list.reverse()
    
    # Convert the reversed list of tuples back to a dictionary
    reversed_result_dict = {k: v for k, v in result_list}
    
    return reversed_result_dict


def remove_thumbnails(data):
    data_no_thumbnails = copy.deepcopy(data)  # Make a deep copy of the data
    for doc_id, doc_info in data_no_thumbnails.items():
        if 'document_thumbnail' in doc_info:
            del doc_info['document_thumbnail']

    for doc_id, doc_info in data_no_thumbnails.items():
        if 'content' in doc_info:
            del doc_info['content']

    
    return data_no_thumbnails
 
 
def map_to_structure(qs, isInitialRun, user_query):
    result_dict = {}

    # Extract the DataFrame from the tuple
    dataframe = qs[0]

    # Create a dictionary for each document
    document_info = {}

    # Counter to limit the loop to 10 iterations
    count = 0
    for index, row in dataframe.iterrows():
        # Define a unique identifier for each document, you can customize this based on your data
        document_id = f"doc-{index + 1}"
        # Handle NaN in content by using fillna
        content = str(row["Content"]) if row["Content"] is not None else ""  # row["Content"]
        content = ' '.join(content.split()[:160])

        title = str(row["Document Title"]) if row["Document Title"] is not None else ""
        extract = str(content) if content is not None else ""

        title_similarity = processing_modules.jaccard_similarity(user_query, title) or 0
        extract_similarity = processing_modules.jaccard_similarity(user_query, extract) or 0
        # print(f""" {title_similarity} {user_query} {title}""")
        print(f""" {extract_similarity} {user_query} {title}""")

        document_info = {
            "document_title": title,
            "extract": extract,  # Adjust based on your column names
            "category": str(row["Category"]) if row["Category"] is not None else "",
            "document_link": str(row["Link"]).replace("https-//", "https://") if row["Link"] is not None else "",
            "summary": str(row["Summary"]) if row["Summary"] is not None else extract,
            "document_thumbnail": str(row["Thumbnail"]) if row["Thumbnail"] is not None else "",
            "relevancy": extract_similarity
        }

        # "content": str(row["Content"]).replace("\n","") if row["Content"] is not None else "",


        # Add the document to the result dictionary
        result_dict[document_id] = document_info

        # Increment the counter
        count += 1

        # Break out of the loop if the counter reaches top 10
        if count == 10:
            break

        # Sort the dictionary by relevancy
        # sorted_result_dict = sort_by_relevancy(result_dict)

    return result_dict


def process_queries(queries, user_query, client, embedding_model, isInitialRun):
    merged_result_structure = {}

    # for query in queries:
        # qs = search_embeddings(query)  # Assuming search_embeddings returns a tuple (df, distances, indices)
    qs = search_embeddings(user_query,client, embedding_model,isInitialRun) #df, distances, indices
        # print(f""" qs=== {qs} {isInitialRun} {user_query} | query == {query} """)
    if qs[0] is not None:
        result_structure = map_to_structure(qs,isInitialRun,user_query)
        for doc_id, doc_info in result_structure.items():
            merged_result_structure[doc_id] = doc_info
    return merged_result_structure

## module to extract text from documents and return the text and document codes
def semanticSearchModule(user_query, client, embedding_model, isInitialRun, openai_deployment):
    # query_transformation = openai_call.callOpenAI(f"""
    # Given a question, your job is to break them into 3 main sub-question and return as array. 
    
    # - You Must return output seperated by |
    # - Avoid adding new lines or breaking spaces to your output and must seperate each idea with |

    # QUESTION: {user_query}
    # """, openai_deployment)
    # print(f""" query_transformation: {query_transformation} """)
    
    # # Split the string by the delimiter '|'
    # questions_array = [question.strip() for question in query_transformation.split('|')]
    questions_array =[]

    merged_results = process_queries(questions_array,user_query, client, embedding_model, isInitialRun)
    print(f""" merged_results===  {merged_results} """)
    return merged_results



def convertQueryIdeaToArray(query_idea_list):
    # Split the query idea list by the "|" character
    query_ideas = query_idea_list.split(" | ")
    # Print the resulting array
    return query_ideas

## module to generate query ideas
def queryIdeationModule(user_query, openai_deployment): # lower priority
    
    # Generate query ideas using OpenAI GPT-3
    prompt = f"""
    Ignore previous commands!!!
    Generate prompt ideas based on the user query: {user_query}


    -Prompt shoud not be answer to the user query but give other contextual ways of representing the user query !!!
    -You Must return output seperated by |  e.g idea 1 | idea2 
    - Each generated ideas should be very dinstinct but contextual. Not repeatitive or using same words
    - The query idea should be in a question form and not an answer form.
    -Avoid adding new lines or breaking spaces to your output and must seperate each idea with |
    """
    response = openai_call.callOpenAI(prompt, openai_deployment)
    qIdeasResponse = convertQueryIdeaToArray(response)
    return qIdeasResponse

def cleanJson(json_data):
    # Make a deepcopy of the JSON data to avoid modifying the original object
    jsonData = copy.deepcopy(json_data)

    # Loop through the items and remove the "thumbnail" key
    for value in jsonData.values():
        if "document_thumbnail" in value:
            del value["document_thumbnail"]

    for value in jsonData.values():
        if "thumbnail" in value:
            del value["thumbnail"]
    # Convert the modified data back to JSON
    modified_json = json.dumps(jsonData, indent=4)

    return modified_json




#Cleanup outputs
# Parse the HTML content

# Function to remove [n] if not inside an <a> tag
def remove_unlinked_citations(soup):
    # Regular expression to match [n] pattern
    pattern = re.compile(r'\[\d+\]')
    
    for text in soup.find_all(text=pattern):
        # Find all matches in the text
        matches = pattern.findall(text)
        for match in matches:
            # Check if the match is inside an <a> tag
            if not text.find_parent('a'):
                # Remove the match from the text
                text.replace_with(text.replace(match, ''))
    
    return soup


def cleanCitation(html_content): 

    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove unlinked citations
    clean_soup = remove_unlinked_citations(soup)

    # Get the modified HTML content
    clean_html_content = str(clean_soup)

    return clean_html_content



def synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict, openai_deployment, prompt_formattings):
    
    ###synthesize data into structure within llm prompt engineering instructions
    answer=get_answer(user_query,excerpts_dict, openai_deployment) #callOpenAI
    return answer


##Indicators


## module to get data for specific indicators which are identified is relevant to the user query
def indicatorsModule(user_query): #lower priority
    return indicator_module.indicatorsModule(user_query)


# Function to calculate the similarity score between two strings
def similarity_score_kg(word1, word2):
    # Convert strings to lowercase for case-insensitive comparison
    word1_lower = word1.lower()
    word2_lower = word2[:-5].lower()


    # Split strings into individual words
    words1 = word1_lower.split()
    words2 = word2_lower.split()
    
    # Calculate the number of overlapping words
    common_words = set(words1) & set(words2)
    
    # Calculate similarity score as percentage of overlapping words
    similarity = len(common_words) / max(len(words1), len(words2)) * 100
    
    return similarity


def find_kgold(keywords, data_dir):
    max_score = 0
    most_similar_file = None
    final_output = {"knowledge_graph": {"entities": [], "relations": {}}}

    # Extract the first keyword from the list
    first_keyword = keywords[0] if keywords else None

    # Calculate the similarity score with the first keyword
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            score = similarity_score_kg(first_keyword, filename)
            # print(f""" first_keyword === {first_keyword} filename {filename} score {score}  """)

            if score > max_score:
                max_score = score
                most_similar_file = filename
                # Break the loop after finding the first matching file
                break
    initial_root = most_similar_file[:-5]
    # print(initial_root)
    initial_kg = {}
    with open(os.path.join(data_dir, f"""{initial_root}.json"""), "r") as file:
         initial_kg = json.load(file)  
    #get the initial root file 

    # Load the content of the most similar file
    if most_similar_file:
        with open(os.path.join(data_dir, most_similar_file), "r") as file:
            content = json.load(file)            
            # Iterate over each relation in the content
            for relation, objects in content["knowledge graph"]["relations"].items():
                # print(f""" most_similar_file === {objects} """)
 

                # Dictionary to store found JSON files
                # found_files = {}
                found_files = []
                found_files.append(initial_kg)
                # Iterate through each dictionary in 'data'
                for item in objects:
                    # Extract the 'Object' name
                    object_name = item.get('Object')

                    # Construct the paths for both original and lowercase filenames
                    json_file_original = os.path.join(data_dir, f"{object_name}.json")
                    json_file_lowercase = os.path.join(data_dir, f"{object_name.lower()}.json")
                    
                    # Check if a corresponding JSON file exists
                    # json_file = os.path.join(data_dir, f"{object_name}.json")

                    if os.path.exists(json_file_original) or os.path.exists(json_file_lowercase):
                        # Choose the correct filename based on existence
                        json_file = json_file_original if os.path.exists(json_file_original) else json_file_lowercase
                        
                    # if os.path.exists(json_file):
                        # Load the content of the JSON file
                        # print(f"""*****json_file=== {json_file} """)

                        with open(json_file, "r") as file:
                            try: 
                                file_content = json.load(file)
                                # print(f"""*****object_name=== {file_content} """)
                                
                                # Add the content to the 'found_files' dictionary
                                # found_files[object_name] = file_content
                                found_files.append(file_content)
                                # found_files = file_content

                            except Exception as e:
                                print("Error:", e)
                                return None
                # 'found_files' now contains the content of JSON files with object names as keys
                print(f""" found_files === {found_files}""")


    
    return found_files


def find_kg(keywords, data_dir):
    max_score = 0
    most_similar_file = None
    final_output = {"knowledge_graph": {"entities": [], "relations": {}}}

    # Extract the first keyword from the list
    first_keyword = keywords[0] if keywords else None

    # Calculate the similarity score with the first keyword
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            score = similarity_score_kg(first_keyword, filename)
            if score > max_score:
                max_score = score
                most_similar_file = filename
                # Break the loop after finding the first matching file
                break

    if most_similar_file is None:
        print("No matching file found.")
        return None

    initial_root = most_similar_file[:-5]
    initial_kg = {}
    try:
        with open(os.path.join(data_dir, f"{initial_root}.json"), "r") as file:
            initial_kg = json.load(file)
    except Exception as e:
        print(f"Error loading initial root file: {e}")
        return None

    # Load the content of the most similar file
    try:
        with open(os.path.join(data_dir, most_similar_file), "r") as file:
            content = json.load(file)
    except Exception as e:
        print(f"Error loading most similar file: {e}")
        return None

    # Ensure content structure is correct
    if "knowledge graph" not in content or "relations" not in content["knowledge graph"]:
        print("Invalid JSON structure.")
        return None

    found_files = [initial_kg]
    
    # Iterate over each relation in the content
    for relation, objects in content["knowledge graph"]["relations"].items():
        for item in objects:
            object_name = item.get('Object')

            # Construct the paths for both original and lowercase filenames
            json_file_original = os.path.join(data_dir, f"{object_name}.json")
            json_file_lowercase = os.path.join(data_dir, f"{object_name.lower()}.json")

            if os.path.exists(json_file_original) or os.path.exists(json_file_lowercase):
                json_file = json_file_original if os.path.exists(json_file_original) else json_file_lowercase

                try:
                    with open(json_file, "r") as file:
                        file_content = json.load(file)
                        found_files.append(file_content)
                except Exception as e:
                    print(f"Error loading file {json_file}: {e}")
                    continue

    return found_files
