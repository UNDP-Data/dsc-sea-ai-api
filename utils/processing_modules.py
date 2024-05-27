
from html import entities
import utils.openai_call as openai_call
import openai
import ast
import pandas as pd
import faiss
import numpy as np
import pycountry
import re

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
import utils.indicator as indicator_module


# model = transformers.BertModel.from_pretrained('bert-base-uncased')
# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_pickle('./models/df_embed_EN_All_V3.pkl')

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


'''
Previous 'find_mentioned_countries' can detect countries when they are formed correctly.

'''
# Extract mentioned countries' ISO3 code
def find_mentioned_country_code(user_query):
    countries = set()
    extracted_countries = find_countries(user_query, is_ignore_case=True)
    # check if we have country first
    if extracted_countries:
        for c in extracted_countries:
            countries.add(c[0].alpha_3)
    # check if we have continent
    else:
        words = re.findall(r'\w+|[^\w\s]', user_query)
        text = ' '.join(words)  # Join the tokens back into a string    

        world_info = awoc.AWOC()
        all_continents = set([continent.lower() for continent in world_info.get_continents_list()])
        for word in text.split():
            word = word.lower()
            # check if this continent
            if word in all_continents:
                target_countries = world_info.get_countries_list_of(word)
                for country in target_countries:
                    countries.add(world_info.get_country_data(country)['ISO3'])
    return countries

def filter_country(user_query):
    mentioned_countries = find_mentioned_country_code(user_query)
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
    # print("All Entities and POS:", entities)
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
    #    print(f'on the reference df {filtered_df_backup_reference.empty}')
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
def get_answer(user_question, relevant_docs,openai_deployment): 

    formatting = f""" 
        Strictly follow the follow steps:
        Your answer only in HTML syntax with HTML tags.
        Use HTML tags like < ul>, < ol>, < li>,  < strong>, < p>
        Only consider the inner part of the < body> tag.
        ALWAYS use the following tag for new lines: < br />
        use <a> for cite numbers
        Do not add CSS attributes.
        Your answer must be formatted in HTML format
        
        
        1. <<SOUCRCE>> for citing : {relevant_docs}

        2. Must Follow the Structure for your output strictly: 

            
                Break down the answer into main points.
                Provide explanations and cite <<SOUCRCE>> directly.

                Summarize the main points.

                For Example: 
                Question: What are the benefits of regular exercise?
                
                Answer
                Regular exercise offers significant benefits across physical, mental, and emotional dimensions. Physically, it strengthens the heart and enhances circulation, which can reduce the risk of heart disease by up to 40% <a href=“link” data-id=“doc-id”>[1]</a>. It also aids in weight management by burning calories and building muscle mass <a href=“link” data-id=“doc-id”>[2]</a>.
                Mentally, exercise helps reduce anxiety and depression by increasing the production of endorphins, acting similarly to some medications. Additionally, it enhances cognitive function, improving memory and learning capabilities <a href=“link” data-id=“doc-id”>[3]</a>.
                Emotionally and socially, regular physical activity reduces stress hormones such as cortisol  and boosts self-esteem by improving body image and self-confidence. Participation in group sports also provides social benefits and fosters a sense of community.
                In summary, regular exercise is essential for overall health, enhancing physical fitness, mental clarity, and emotional well-being.


    3. Return answer without explicit partitions, keeping it concise and integrating citations naturally within the text.
    4. You must cite at least one relevant <<SOUCRCE>> provided above. This is compulsory. Don't just show a reference at the footnotes but also integrate into the text using the cite number e.g <a href=“link” data-id=“doc-id”>[1]</a>.
    5. Cited source number must be an anchor tag link !!!!! You don't have to always cite. Only do this when relevant. If you can't find the link just leave the citation out.
    6. VERY IMPORTANT:  avoid numberic listings in your answer!!!!!
    """
   
 
   
    print(f""" excerpts_dict === {len(relevant_docs)} """)

    # print(formatting)
    # print(openai_deployment)

    messages = [
            {"role": "system", "content":"You are a helpful assistant and a professional writer with 50 years experience. Give answer to the user's inquiry."
            },
            {'role': 'user', 'content': f"""{formatting} 
                                            {user_question}"""},
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
        return relabel_and_add_citations(result_structure)
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
        if "document_thumbnail" in value:
            del value["document_thumbnail"]

    for value in jsonData.values():
        if "thumbnail" in value:
            del value["thumbnail"]
    # Convert the modified data back to JSON
    modified_json = json.dumps(jsonData, indent=4)

    return modified_json


# Function to relabel keys and add citations
def relabel_and_add_citations(data):
    new_data = {}
    citation_counter = 1

    for doc_id, doc_info in data.items():
        new_data[doc_id] = {
            "document_title": doc_info.get("title", ""),
            "summary": doc_info.get("extract", ""),
            "document_category": doc_info.get("category", ""),
            "document_link": doc_info.get("link", ""),
            "document_thumbnail": doc_info.get("thumbnail", ""),
            "citation": citation_counter
        }
        citation_counter += 1

    return new_data

# module to synthesize answer using retreival augmented generation approach
def synthesisModuleOLD(user_query, entities_dict, excerpts_dict, indicators_dict, openai_deployment):
    
    excerpts_dict_ = cleanJson(excerpts_dict)
    # print(excerpts_dict_)
    # Generate prompt engineering text and template
    llm_instructions_old = f"""
    Ignore previous commands!!!
    Given a user query, use the provided <Sources> extract section of the JSON only to provide the correct answer to the user's query.

    User Query: {user_query}

    Sources: {excerpts_dict_}
    
    - Answer output must be properly formatted using HTML. 
    - Don't include <html>, <script>, <link> <a>  or <body> tags. Only text formating tags should be allowed. e.g h1..h3, p, anchor, etc. Strictly HTML only
    - You can your answers from the relevant <Sources> also and make citations to Source extract when referenced 
    - The Source as format like: "doc-n": {{
        "title": "title of the relate document",
        "extract": "content",
        "category": "",
        "link": "",
        "thumbnail": "",
        "citation": n
    }}, where doc-n can be doc-1, doc-24 etc.. n is in integer.
    - Reference the extract and title of all document sources provided in the json and summarise it into a coherent answer that relates to the <User Query> when possible
    - Citation should follow formats: [reference content][citation number]. 
    - Give output writing tone like a academic research tone
    - Remove new line or tab characters from your output
    - do not use or include links,  anchor links or a href tags !!!
    - do not include references links at the end or show References!!!
    - to reference within the text do [n] not [source n] . musct be [n] where n is an integer of the citation number
    - Should be one citation only e.g [n] not [n][n][n]
     """

    llm_instructions = f""" 
    User Query: {user_query}
    Sources: {excerpts_dict_}
    Solve by breaking the problem into steps.

     """

    print(llm_instructions)
    ###synthesize data into structure within llm prompt engineering instructions
    answer= openai_call.callOpenAI(llm_instructions, openai_deployment)
    
    return answer.replace("</p>\n\n<p>", "<br/>").replace("</p>\n<p>","<br/>").replace("\n","<br/>")

def synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict,openai_deployment):

    excerpts_dict_ = cleanJson(excerpts_dict)

    ###synthesize data into structure within llm prompt engineering instructions
    answer=get_answer(user_query, excerpts_dict_,openai_deployment) #callOpenAI
    answer_formated_fixed = answer.replace("\n\n","<br>").replace("\n","<br>")
    return answer_formated_fixed

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

                # Iterate over each object in the relation
                # for obj in objects:
                #     # Check if the object has a "Object" key
                #     if "Object" in obj:
                #         # Search for files with the same name as the object
                #         obj_filename = f"{obj['Object']}.json"
                #         # Check if the file path exists
                #         full_path = os.path.join(data_dir, obj_filename)
                        
                #         if os.path.exists(full_path):
                #             # Load the content of the object file
                #             with open(full_path, "r") as obj_file:
                #                 obj_content = json.load(obj_file)
                #                 # print(f""" objects====**** {obj_content}""")

                #                 # Merge the content of the object file into the final output
                #                 final_output["knowledge_graph"]["relations"].setdefault(relation, []).append(obj_content)
                #         else:
                #             # File path does not exist
                #             error = ''
                #             # print(f"The file path {full_path} does not exist.")

    # Merge relations into a single JSON
    # merged_relations = {}
    # for relations_list in final_output["knowledge_graph"]["relations"].values():
    #     for relation_item in relations_list:
    #         # Check if the "Object" key exists in relation_item
    #         relation_name = relation_item.get("Object", "")
    #         merged_relations.setdefault(relation_name, []).append(relation_item)
    # # print(f""" ****merged_relations***#### == {merged_relations} """)
    # final_output["knowledge_graph"]["relations"] = merged_relations

    
    return found_files

