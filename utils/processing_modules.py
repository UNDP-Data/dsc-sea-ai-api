
import utils.openai_call as openai_call
import openai
import ast
import pandas as pd
import faiss
import numpy as np
import pycountry
import re
import transformers
import torch
from sklearn.metrics.pairwise import cosine_similarity


model = transformers.BertModel.from_pretrained('bert-base-uncased')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_pickle('./models/df_embed_EN_All.pkl')

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
        entity_info = openai_call.callOpenAI(prompt, openai_deployment)
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


#Context similarity
def contextualSimilarity(user_query,df_title) :

    user_query = str(user_query)
    df_title = str(df_title)
    # Tokenize and encode the sentences
    tokens1 = tokenizer(user_query, return_tensors='pt', max_length=512, truncation=True)
    tokens2 = tokenizer(df_title, return_tensors='pt', max_length=512, truncation=True)

    # Forward pass through the BERT model
    with torch.no_grad():
        output1 = model(**tokens1)
        output2 = model(**tokens2)

    # Get the embeddings (CLS token)
    embedding1 = output1.last_hidden_state[:, 0, :]
    embedding2 = output2.last_hidden_state[:, 0, :]

    # Calculate cosine similarity
    context_similarity = cosine_similarity(embedding1, embedding2)
    return context_similarity[0][0]
 


#This contains all filters for the semantic search
#Context Similarity takes two queries and find how similar they are "context wise"
#E.g "My house is empty today" and "Nobody is at my home" are same context but not word similarity
# - Filter country relevant documents when mentioned 
# - Filter by Context similarity in user_query and title, journal, content etc.

def filter_semantics(user_query):
    mentioned_countries = find_mentioned_countries(user_query)
    
    if mentioned_countries:
        country = mentioned_countries[0]
        filtered_df = df[df['Country Name'] == country]
    else:
        # Calculate contextual similarity for each document title
        similarity_scores = []
        titles_with_scores = []  # Keep track of titles with scores
        
        # Track the number of similarity scores collected
        count = 0
        
        for title in df['Document Title']:
            if title is not None:
                similarity_score = contextualSimilarity(user_query, title)
                if similarity_score > 0.8:
                    similarity_scores.append(similarity_score)
                    titles_with_scores.append(title)
                    count += 1
                    if count >= 10:  # Break once 10 scores are collected
                        break
        
        # Check if any titles have similarity scores
        if not similarity_scores:
            print("No documents found with a similarity score greater than 0.85.")
            return None
        
        # Create a DataFrame to store the similarity scores
        similarity_df = pd.DataFrame({'Document Title': titles_with_scores, 'Similarity Score': similarity_scores})
        
        # Sort the DataFrame by the similarity scores in descending order
        similarity_df = similarity_df.sort_values(by='Similarity Score', ascending=False)
        
        # Return the top 10 results
        filtered_df = df[df['Document Title'].isin(similarity_df['Document Title'])]

    return filtered_df

#run search on the vector pkl embeddings
def search_embeddings(user_query, client, embedding_model):
    df_filtered = filter_semantics(user_query) if filter_semantics(user_query) is not None else None
    
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
def map_to_structure(qs):
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
        # Create a dictionary for each document
        document_info = {
            "title": row["Document Title"],
            "extract": content or "",  # You may need to adjust this based on your column names
            "category": row["Category"],
            "link": row["Link"],
            "thumbnail": ''
        }
        # print(document_info)
        # Add the document to the result dictionary
        result_dict[document_id] = document_info

        # Increment the counter
        count += 1

        # # Break out of the loop if the counter reaches top 10
        if count == 10:
            break

    return result_dict

## module to extract text from documents and return the text and document codes
def semanticSearchModule(user_query, client, embedding_model):
    qs = search_embeddings(user_query,client, embedding_model) #df, distances, indices
    # if qs != None :
    if qs[0] is not None:
        result_structure = map_to_structure(qs)
        return result_structure
    else : 
        return []

## module to generate query ideas
def queryIdeationModule(user_query, openai_deployment): # lower priority
    
    # Generate query ideas using OpenAI GPT-3
    prompt = f"""Generate query ideas based on the user query: {user_query}
    
    -You Must return output in array format e.g ['idea 1', 'idea2'] !!!
    -Avoid adding new lines or breaking spaces to your output. Array should be single dimension and single line !!!
    
    """
    response = openai_call.callOpenAI(prompt, openai_deployment)
    return response


# module to synthesize answer using retreival augmented generation approach
def synthesisModule(user_query, entities_dict, excerpts_dict, indicators_dict, openai_deployment):
    
    # Generate prompt engineering text and template
    llm_instructions = f"""
    Ignore previous commands!!!
    Given a user query, use the provided excerpts, Sources, and entity and relation info to
    provide the correct answer to the user's query
    
    User Query: {user_query}
    
    Sources: {excerpts_dict}
    
    Entity and Relation info: {entities_dict}

    - Answer output must be properly formatted using HTML. 
    - Don't include <html>, <script>, <link> or <body> tags. Only text formating tags should be allowed. e.g h1..h3, p, anchor, etc.
    - Make sure to Include citations based on the Sources. e.g Text excerpt here<a data-id='test-doc-1'>[1]</a> when referencing a document in the sources. using 1 ...nth
    - The citations anchor be single and not follow each other.
    - Use the anchor tag for the citation links and should link to the document link. 
      For example Undp operates in afganistan <a data-id='test-doc-1'>[1]</a>. UNDP offers health relationships <a data-id='test-doc-2'>[2]</a>.
    - Use MUST one citation per excerots. Don't list multiple anchors side by side for citation !!!
    - The text in the anchor tag should be citation number not document title.
    - You can reference multitple citations based sources
    """
    ###synthesize data into structure within llm prompt engineering instructions
    answer= openai_call.callOpenAI(llm_instructions, openai_deployment)
    
    return answer

## module to get data for specific indicators which are identified is relevant to the user query
def indicatorsModule(user_query): #lower priority
    
    # find relevant indicators based on uesr query and extract values
    indicators_dict={
        "indicator-id-1":"value from indicator-id-1",
        "indicator-id-2":"value from indicator-id-2"
    }#temp
    
    return indicators_dict

