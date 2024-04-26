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
