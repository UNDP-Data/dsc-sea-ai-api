
from html import entities
import utils.openai_call as openai_call
import openai
import ast
import pandas as pd
import faiss
import numpy as np
import pycountry
import re

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

# model = transformers.BertModel.from_pretrained('bert-base-uncased')
# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_pickle('./models/df_embed_EN_All_V2.pkl')

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
    
    for word in text.split():
        try:
            country = pycountry.countries.get(name=word) #pycountry.countries.lookup(word)
            if country != None : 
               countries.add(country.name)
        except LookupError:
            pass
    
    return list(countries)

def filter_country(user_query):
    mentioned_countries = find_mentioned_countries(user_query)
    # print(mentioned_countries)
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
def average_word_embedding(sentence):
    # Parse the sentence using SpaCy
    doc = nlp(sentence)
    # Get word vectors and average them
    word_vectors = [token.vector for token in doc if token.has_vector]
    if not word_vectors:
        return None
    return np.mean(word_vectors, axis=0)

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



#This contains all filters for the semantic search
#Context Similarity takes two queries and find how similar they are "context wise"
#E.g "My house is empty today" and "Nobody is at my home" are same context but not word similarity
# - Filter country relevant documents when mentioned 
# - Filter by Context similarity in user_query and title, journal, content etc.

def filter_semantics(user_query, isInitialRun):
    doc = nlp(user_query)
    # # Extract all entities
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != ""]  # Filter out empty entities
    entities.extend((token.text, "NOUN") for token in doc if token.pos_ in ["NOUN","PROPN", "PRON", "PROPN", "NUM", "SYM", "X","ABBR"] or token.is_alpha)

    # # Remove stop words
    entities = [(entity, label) for entity, label in entities if entity.lower() not in STOP_WORDS]
    
    # entities = [('the Country Program Document', 'ORG'), ('Afghanistan', 'GPE'), ('the year 2014', 'DATE'), ('Afghanistan', 'GPE'), ('UNDP', 'ORG'), ('2015-2019', 'DATE'), ('looking', 'NOUN'), ('insights', 'NOUN'), ('Country', 'NOUN'), ('Program', 'NOUN'), ('Document', 'NOUN'), ('Afghanistan', 'NOUN'), ('year', 'NOUN'), ('2014', 'NOUN'), ('particularly', 'NOUN'), ('interested', 'NOUN'), ('understanding', 'NOUN'), ('Afghanistan', 'NOUN'), ('strategies', 'NOUN'), ('related', 'NOUN'), ('economic', 'NOUN'), ('development', 'NOUN'), ('governance', 'NOUN'), ('social', 'NOUN'), ('inclusion', 'NOUN'), ('Additionally', 'NOUN'), ('like', 'NOUN'), ('know', 'NOUN'), ('partnerships', 'NOUN'), ('international', 'NOUN'), ('organizations', 'NOUN'), ('UNDP', 'NOUN'), ('poverty', 'NOUN'), ('reduction', 'NOUN'), ('initiatives', 'NOUN'), ('gender', 'NOUN'), ('equality', 'NOUN'), ('measures', 'NOUN'), ('included', 'NOUN'), ('program', 'NOUN'), ('provide', 'NOUN'), ('details', 'NOUN'), ('planned', 'NOUN'), ('address', 'NOUN'), ('security', 'NOUN'), ('issues', 'NOUN'), ('sustainable', 'NOUN'), ('development', 'NOUN'), ('goals', 'NOUN'), ('timeframe', 'NOUN'), ('2015', 'NOUN'), ('-', 'NOUN'), ('2019', 'NOUN')]
    # Print the extracted entities
    print("All Entities and POS:", entities)
    # Generate DFs for main entities
    filtered_df_country = pd.DataFrame()  # Initialize an empty DataFrame
    filtered_df_others = pd.DataFrame()  # Initialize an empty DataFrame
    filtered_df_backup_reference = pd.DataFrame() # Initialize an empty DataFrame
    allow_low = True

    # START 
    for entity, label in entities:
        # print(entity)
        filtered_df_others = pd.concat([filtered_df_others, df[df['Document Title'].str.lower().str.contains(entity.lower(), na=False)]])

        #Calculate similarity scores for each document title
        similarity_scores = []
        document_titles = []
        similarity_score = 0
        
        # Iterate through each document title and calculate similarity score
        for title in filtered_df_others['Document Title']:
            if title is not None:

                # if isInitialRun : 
                #     similarity_score = jaccard_similarity(user_query,title)   
                # else :
                #     similarity_score = calculate_context_similarity(user_query,title)   
                similarity_score = calculate_context_similarity(user_query,title)   

                similarity_scores.append(similarity_score)
                document_titles.append(title)
        
        # Create DataFrame only with valid similarity scores
        similarity_df = pd.DataFrame({'Document Title': document_titles, 'Similarity Score': similarity_scores})
        
        df_temp = pd.concat([df])
        # threshold = 0
        # if isInitialRun : 
        #     threshold = 0.05
        # else : 
        #     threshold = 0.5
        threshold = 0.5


        # Filter df based on similarity scores greater than threshold for filtered_df_others
        filtered_df_others = df[df['Document Title'].isin(similarity_df[similarity_df['Similarity Score'] > threshold]['Document Title'])]
        filtered_df_backup_reference = pd.concat([filtered_df_backup_reference,  df_temp[df_temp['Document Title'].isin(similarity_df[(similarity_df['Similarity Score'] >= 0.28) & (similarity_df['Similarity Score'] < 0.45)]['Document Title'])] ])
        
        #Check for location related e.g by country, language, locals
        if label in ['GPE', 'NORP', 'LANGUAGE', 'FAC']:
            filtered_df_country = pd.concat([filtered_df_country, df[df['Country Name'] == entity]])
   
    # END



    merged_df = pd.DataFrame()
    if filtered_df_others.empty and filtered_df_country.empty:
       print(f'on the reference df {filtered_df_backup_reference.empty}')
       merged_df = pd.concat([filtered_df_backup_reference])
    else :
       merged_df = pd.concat([filtered_df_country,filtered_df_others])
    
    return merged_df


 
#run search on the vector pkl embeddings
def search_embeddings(user_query, client, embedding_model, isInitialRun):
    # df_filtered = filter_semantics(user_query) if filter_semantics(user_query) is not None else None
    # Call filter_semantics function once and store the result in a variable
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
def get_answer(user_question, content, openai_deployment):
    system_prompt = "You are a system that answers user questions based on excerpts from PDF documents provided for context. Only answer if the answer can be found in the provided context. Do not make up the answer; if you cannot find the answer, say so."
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question},
        {'role': 'user', 'content': content},
    ]
    response_entities = openai.chat.completions.create(
                    model=openai_deployment,
                    temperature=0.2,
                    messages=messages
                )
    response = response_entities.choices[0].message.content
    return response
  
# map to structure
def map_to_structure(qs, isInitialRun):
    result_dict = {}

    # Extract the DataFrame from the tuple
    dataframe = qs[0]

    # Counter to limit the loop to 10 iterations
    count = 0

    for index, row in dataframe.iterrows():
        # Define a unique identifier for each document, you can customize this based on your data
        document_id = f"doc-{index + 1}"
        # Handle NaN in content by using fillna
        content = row["Content"]
        content = ' '.join(row["Content"].split()[:160])
        doc_url = row["Link"]
        thumbnail_base64 = '' #generate_thumbnail_from_pdf(doc_url)

        # Create a dictionary for each document
        document_info = {
            "title": row["Document Title"],
            "extract": content or "",  # You may need to adjust this based on your column names
            "category": row["Category"],
            "link": doc_url,
            "thumbnail": row["Thumbnail"]
        }
        # print(document_info)
        # Add the document to the result dictionary
        result_dict[document_id] = document_info

        # Increment the counter
        count += 1

     
        if isInitialRun and count == 1:
            break
        #top k-5 docs only
        elif not isInitialRun and count == 10:  
            break

    
    return result_dict

## module to extract text from documents and return the text and document codes
def semanticSearchModule(user_query, client, embedding_model, isInitialRun):
    qs = search_embeddings(user_query,client, embedding_model,isInitialRun) #df, distances, indices
    # if qs != None :
    if qs[0] is not None:
        result_structure = map_to_structure(qs,isInitialRun)
        return result_structure
    else : 
        return []

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
        if "thumbnail" in value:
            del value["thumbnail"]

    # Convert the modified data back to JSON
    modified_json = json.dumps(jsonData, indent=4)

    return modified_json



# module to synthesize answer using retreival augmented generation approach
def synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict, openai_deployment):
    
    excerpts_dict_ = cleanJson(excerpts_dict)
    # print(excerpts_dict_)
    # Generate prompt engineering text and template
    llm_instructions = f"""
    Ignore previous commands!!!
    Given a user query, use the provided <Sources> extract section of the JSON only to provide the correct answer to the user's query.

    User Query: {user_query}

    Sources: {excerpts_dict_}
    
    - Answer output must be properly formatted using HTML. 
    - Don't include <html>, <script>, <link> or <body> tags. Only text formating tags should be allowed. e.g h1..h3, p, anchor, etc. Strictly HTML only
    - Strictly infer your answers from the <Sources> Only and make citations to Source extract referenced 
    - The Source as format like: "doc-n": {{
        "title": "title of the relate document",
        "extract": "content",
        "category": "",
        "link": "",
        "thumbnail": ""
    }}, where doc-n can be doc-1, doc-24 etc.. n is in integer.
    - Reference the extract and title of all document sources provided in the json and summarise it into a coherent answer that relates to the <User Query>
    - Citation should follow formats: [reference content]<a href='link here' data-id='doc-n'>[i]</a> . The reference bracket should be the reference link
    - Give output writing tone like a academic research tone
    - Strictly use IEEE Citation Style 
    - If no <Sources> are provided, try to make suggestives or  simply say you don't have that information   
    - Remove new line or tab characters from your output
        
    """
    ###synthesize data into structure within llm prompt engineering instructions
    answer= openai_call.callOpenAI(llm_instructions, openai_deployment)
    
    return answer.replace("</p>\n\n<p>", "<br/>").replace("</p>\n<p>","<br/>").replace("\n","<br/>")

## module to get data for specific indicators which are identified is relevant to the user query
def indicatorsModule(user_query): #lower priority
    
    # find relevant indicators based on uesr query and extract values
    indicators_dict={
        "indicator-id-1":"value from indicator-id-1",
        "indicator-id-2":"value from indicator-id-2"
    }#temp
    
    return indicators_dict

