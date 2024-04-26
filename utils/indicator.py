import numpy as np
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import re
import tiktoken
import time
import faiss
import awoc
import spacy
import csv
nlp = spacy.load("en_core_web_sm")
from sklearn.metrics.pairwise import cosine_similarity
from sutime import SUTime
import json
from datetime import datetime
from bert_score import score as bert_score
from country_named_entity_recognition import find_countries
import utils.processing_modules as processing_modules

# import sys
# sys.path.insert(1, '../utils')

# from processing_modules_for_test_indicator import semanticSearchModule

wdi_csv = pd.read_csv('../data/WDI_CSV/WDICSV.csv')
# country meta data
wdi_country = pd.read_csv('../data/WDI_CSV/WDICountry.csv')
# Series meta data
wdi_series = pd.read_csv('../data/WDI_CSV/WDISeries.csv')


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

encoding = tiktoken.get_encoding('cl100k_base')
embedding_model = os.getenv("USER_QUERY_EMBEDDING_ENGINE") 

df = pd.read_pickle('../data/indicator_meta_embed.pkl')

load_dotenv()

def create_embedding(row):
    time.sleep(3)
    #print(row.name)
    input_text = row['Indicator Name'].replace("\n", " ")
    input_text = re.sub(r'\s+', ' ', input_text)
    encodings = encoding.encode(input_text)
    length = len(encodings)
    embedding = client.embeddings.create( 
        input=input_text ,model= embedding_model
    ).data[0].embedding
    
    return length, embedding


def filter_indicators(user_query):
    # Calculate similarity scores for each indicators
    similarity_scores = []
    indicators = []

    # Iterate through each indicator title and calculate similarity score
    for indicator in df['Indicator Name']:
        similarity_score = processing_modules.jaccard_similarity(user_query, indicator)
        similarity_scores.append(similarity_score)
        indicators.append(indicator)
        
    # Create DataFrame only with valid similarity scores
    similarity_df = pd.DataFrame({'Indicator Name': indicators, 'Similarity Score': similarity_scores})
    similarity_df = similarity_df.sort_values('Similarity Score', ascending=False)
    similarity_df = similarity_df[:10]
        
    # Filter indicators where similarity score is above a threshold (e.g., 0.3)
    threshold = 0.01
    filtered_df = df[df['Indicator Name'].isin(similarity_df[similarity_df['Similarity Score'] > threshold]['Indicator Name'])]

    return  list(filtered_df['Series Code'])
 

# search target indicator
# Implement this function later
def search_embeddings(user_query):
    df_filtered = filter_indicators(user_query) if filter_indicators(user_query) is not None else None
    
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


# Extract set of years from given timex3_list
def timex3_to_year_list(timex3_list):
    year_list = set()
    for timex3 in timex3_list:
        sutimeType, value = timex3["type"], timex3["value"]
        if "REF" not in value:
            if isinstance(value, dict):
                if value:
                    for year in range(int(value['begin']), int(value['end']) + 1):
                        year_list.add(str(year))
            elif value.isdigit():
                year_list.add(str(value))
            elif sutimeType in ['DATE', 'DURATION']:
                if sutimeType == 'DATE':
                    res = re.search('^\d\d\d\d', value)
                    if res:
                        year_list.add(str(res.group(0)))
                else:
                    year_dur = 0
                    current_year = datetime.now().year
                    dur_list = re.findall('\d+', "".join(re.findall('P[0-9]+Y', value)))
                    if dur_list:
                        year_dur = max([int(y) for y in dur_list])
                        while year_dur:
                            year_list.add(str(current_year - year_dur))
                            year_dur -= 1
            else:
                continue
    return list(year_list)


def find_target_period(user_query):
    sutime = SUTime(mark_time_ranges = True, include_range = True)
    res = sutime.parse(user_query)
    return timex3_to_year_list(res)

def map_to_structure(countries, indicators, years):
    # load all indicator dataset
    # wdi_csv = pd.read_csv('../data/WDI_CSV/WDICSV.csv')
    count = 0
    result_dict = {}
    for country in countries:
        for indicator in indicators:
            indicator_id = f"indicator-{count + 1}"
            target_row = wdi_csv[(wdi_csv['Country Code'] == country) & (wdi_csv['Indicator Code'] == indicator)]
            if not target_row.empty:
                country_name, indicator_name = target_row['Country Name'].values[0], target_row['Indicator Name'].values[0]
                if years:
                    target_row = target_row[years]
                else:
                    target_row = target_row.iloc[:,4:]
                target_row = target_row.dropna(axis=1)
                if not target_row.empty:
                    year_to_value = {}
                    for column in target_row:
                        year_to_value[column] = target_row[column].values[0]
                    indicator_info = {
                        "Country": country_name,
                        "Indicator Name": indicator_name,
                        "Values Per Year": year_to_value
                    }
                    
                    result_dict[indicator_id] = indicator_info
                    # Increment the counter
                    count += 1
        if count == 30:
            break
    return result_dict


## module to extract text from documents and return the text and document codes
def indicatorsModule(user_query):
    countries = processing_modules.find_mentioned_country_code(user_query)
    indicators = filter_indicators(user_query) #df, distances, indices
    years = find_target_period(user_query)
    if countries and indicators:
        # Reduce Indicator List to 2 if countries are too many
        if len(countries) > 5:
            indicators = indicators[:2]
        # for testing
        #result_structure = {}
        #result_structure["User Query"] = user_query
        #result_structure["indicatorsModule Result"] = map_to_structure(countries, indicators, years)
        result_structure = map_to_structure(countries, indicators, years)
        return result_structure
    else:
        return {}

