import ast
import copy
import re

import faiss
import numpy as np
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import genai, storage

df = storage.read_json("models/df_embed_EN_All_V4.jsonl", lines=True)


# Extract entities for the query and return the extract entities as an array
def extract_entities(user_query, openai_deployment):
    prompt = f"""
    Extract entities from the following user query: \"{user_query}\" and return output in array format.
    
    -Entities should be directly related to the domain or topic of interest. They should represent important concepts that contribute to the understanding of the subject matter.
    -Each entity in the knowledge graph should be distinct and have a unique identifier. This ensures clarity and avoids ambiguity when establishing relationships between entities.
    -You Must return output in array format e.g  ['entity1','entity2'] !!!
    -Avoid adding new lines or breaking spaces to your output. Array should be single dimension and single line !!!
 
    """
    entity_list = genai.generate_response(prompt, openai_deployment)
    return entity_list


## module to get information on the entities from user query using the KG
def get_knowledge_graph(user_query, openai_deployment):

    # generate list of entities based on user query
    entity_list = extract_entities(user_query, openai_deployment)
    my_list = ast.literal_eval(entity_list)
    prompt_summarise_entites = f"""
    Summarize all relations between all the entities : {my_list}
    """
    summarise_entities = genai.generate_response(
        prompt_summarise_entites, openai_deployment
    )
    # Initialize an empty dictionary to store information
    entities_dict = {"relations": summarise_entities, "entities": {}}
    # Loop through each entity in the list
    for entity in my_list:
        # Fetch information about the entity from your knowledge graph
        prompt = f"Give me a short description 50 words of {entity}"
        entity_info = ""  # openai_call.generate_response(prompt, openai_deployment)
        # Add the entity information to the dictionary
        entities_dict["entities"][entity] = entity_info

    return entities_dict


# Function to calculate Jaccard similarity between two texts
def jaccard_similarity(text1, text2):
    # Tokenize texts
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0


def filter_semantics(user_query, isInitialRun):
    # If no documents match the keyword, return an empty DataFrame
    if df.empty:
        return pd.DataFrame()

    # Concatenate 'Document Title' and 'Country Name' to form the text for each document
    df["combined_text"] = (
        df["Document Title"].astype(str) + " " + df["Country Name"].astype(str)
    )

    # Use TF-IDF Vectorizer to encode the text
    vectorizer = TfidfVectorizer(stop_words="english")
    document_embeddings = vectorizer.fit_transform(df["combined_text"])
    query_embedding = vectorizer.transform([user_query])

    # Calculate cosine similarity between the query and document embeddings
    similarity_scores = cosine_similarity(
        query_embedding, document_embeddings
    ).flatten()

    # Add the similarity scores to the DataFrame
    df["similarity_score"] = similarity_scores

    # Filter the DataFrame to include only documents with a similarity score above 0.6
    filtered_df = df[df["similarity_score"] > 0.5]

    # If the filtered DataFrame is empty, relax the threshold
    if filtered_df.empty:
        filtered_df = df[df["similarity_score"] > 0.2]

    # Sort the filtered DataFrame by similarity score
    filtered_df = filtered_df.sort_values(by="similarity_score", ascending=False)

    return filtered_df


def search_embeddings(user_query, client, embedding_model, isInitialRun):
    filtered_result = filter_semantics(user_query, isInitialRun)
    # Check if the result is not None before assigning it to df_filtered
    df_filtered = filtered_result if filtered_result is not None else None

    if (
        df_filtered is not None and not df_filtered.empty
    ):  # Check if DataFrame is not None and not empty
        length = len(df_filtered.head())
        filtered_embeddings_arrays = np.array(list(df_filtered["Embedding"]))
        index = faiss.IndexFlatIP(filtered_embeddings_arrays.shape[1])
        index.add(filtered_embeddings_arrays)

        user_query_embedding = (
            client.embeddings.create(input=user_query, model=embedding_model)
            .data[0]
            .embedding
        )

        k = min(5, length)
        distances, indices = index.search(np.array([user_query_embedding]), k)
        return df_filtered, distances, indices
    else:
        return None, None, None


# get answer
def get_answer(user_question, relevant_docs, openai_deployment):

    formattings_html = f""" 
        Ignore previous
        Strictly follow the follow steps:
        Your output answer shoud be  in HTML syntax with HTML tags.
        Use HTML tags like < ul>, < ol>, < li>,  < strong>, < p>
        Only consider the inner part of the < body> tag.
        ALWAYS use the following tag for new lines: < br />
        Do not add CSS attributes.
        Include links in the references at the bottom of your answer !!!
        Your final answer must be formatted in HTML format !!!

        Use the extract and summary values to answer the question and reference the document used in links at the bottom. 
        use IEEE (Institute of Electrical and Electronics Engineers) style for referencing.

    """
    formattings = f""" 
        You can use relevant information in the docs to answer also: 

        DOCS: {relevant_docs}
        
       """
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant and a professional researcher with many years of experience in answering questions. Give answer to the user's inquiry. {formattings_html}""",
        },
        {
            "role": "user",
            "content": f"""{formattings} 
                                        {user_question}
                                        
                                         {formattings_html}
                                        """,
        },
    ]

    response_entities = openai.chat.completions.create(
        model=openai_deployment,
        temperature=0.3,
        messages=messages,
        top_p=0.8,
        frequency_penalty=0.6,
        presence_penalty=0.8,
    )
    response = response_entities.choices[0].message.content
    print(f"""cleaned_text {response}""")

    # Define the regex pattern to match digits followed by '. do'
    pattern = r"\d+\. do"

    # Remove matches from the text
    cleaned_text = re.sub(pattern, "", response)

    # # Optionally, clean up any extra spaces or punctuation left behind
    # cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()

    return cleaned_text


def remove_thumbnails(data):
    data_no_thumbnails = copy.deepcopy(data)  # Make a deep copy of the data
    for doc_id, doc_info in data_no_thumbnails.items():
        if "document_thumbnail" in doc_info:
            del doc_info["document_thumbnail"]

    for doc_id, doc_info in data_no_thumbnails.items():
        if "content" in doc_info:
            del doc_info["content"]

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
        content = (
            str(row["Content"]) if row["Content"] is not None else ""
        )  # row["Content"]
        content = " ".join(content.split()[:160])

        title = str(row["Document Title"]) if row["Document Title"] is not None else ""
        extract = str(content) if content is not None else ""

        extract_similarity = jaccard_similarity(user_query, extract) or 0
        print(f""" {extract_similarity} {user_query} {title}""")

        document_info = {
            "document_title": title,
            "extract": extract,  # Adjust based on your column names
            "category": str(row["Category"]) if row["Category"] is not None else "",
            "document_link": (
                str(row["Link"]).replace("https-//", "https://")
                if row["Link"] is not None
                else ""
            ),
            "summary": str(row["Summary"]) if row["Summary"] is not None else extract,
            "document_thumbnail": (
                str(row["Thumbnail"]) if row["Thumbnail"] is not None else ""
            ),
            "relevancy": extract_similarity,
        }

        # "content": str(row["Content"]).replace("\n","") if row["Content"] is not None else "",

        # Add the document to the result dictionary
        result_dict[document_id] = document_info

        # Increment the counter
        count += 1

        # Break out of the loop if the counter reaches top 10
        if count == 10:
            break

    return result_dict


def process_queries(queries, user_query, client, embedding_model, isInitialRun):
    merged_result_structure = {}

    # for query in queries:
    qs = search_embeddings(user_query, client, embedding_model, isInitialRun)
    if qs[0] is not None:
        result_structure = map_to_structure(qs, isInitialRun, user_query)
        for doc_id, doc_info in result_structure.items():
            merged_result_structure[doc_id] = doc_info
    return merged_result_structure


## module to extract text from documents and return the text and document codes
def run_semantic_search(
    user_query, client, embedding_model, isInitialRun, openai_deployment
):
    # query_transformation = openai_call.generate_response(f"""
    # Given a question, your job is to break them into 3 main sub-question and return as array.

    # - You Must return output seperated by |
    # - Avoid adding new lines or breaking spaces to your output and must seperate each idea with |

    # QUESTION: {user_query}
    # """, openai_deployment)
    # print(f""" query_transformation: {query_transformation} """)

    # # Split the string by the delimiter '|'
    # questions_array = [question.strip() for question in query_transformation.split('|')]
    questions_array = []

    merged_results = process_queries(
        questions_array, user_query, client, embedding_model, isInitialRun
    )
    print(f""" merged_results===  {merged_results} """)
    return merged_results


def convert_query_idea_to_array(query_idea_list):
    # Split the query idea list by the "|" character
    query_ideas = query_idea_list.split(" | ")
    # Print the resulting array
    return query_ideas


## module to generate query ideas
def generate_query_ideas(user_query, openai_deployment):  # lower priority

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
    response = genai.generate_response(prompt, openai_deployment)
    qIdeasResponse = convert_query_idea_to_array(response)
    return qIdeasResponse


def get_synthesis(
    user_query,
    entities_dict,
    excerpts_dict,
    indicators_dict,
    openai_deployment,
    prompt_formattings,
):

    ###synthesize data into structure within llm prompt engineering instructions
    answer = get_answer(user_query, excerpts_dict, openai_deployment)
    return answer


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


def find_kg(keywords):
    data_dir = "KG"
    max_score = 0
    most_similar_file = None
    final_output = {"knowledge_graph": {"entities": [], "relations": {}}}

    # Extract the first keyword from the list
    first_keyword = keywords[0] if keywords else None

    # Calculate the similarity score with the first keyword
    client = storage.get_blob_client()
    for filename in client.find(f"{storage.CONTAINER_NAME}/{data_dir}"):
        if filename.endswith(".json"):
            filename = filename.split("/")[-1]
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
        initial_kg = storage.load_json(f"{data_dir}/{initial_root}.json")
    except Exception as e:
        print(f"Error loading initial root file: {e}")
        return None

    # Load the content of the most similar file
    try:
        content = storage.load_json(f"{data_dir}/{most_similar_file}")
    except Exception as e:
        print(f"Error loading most similar file: {e}")
        return None

    # Ensure content structure is correct
    if (
        "knowledge graph" not in content
        or "relations" not in content["knowledge graph"]
    ):
        print("Invalid JSON structure.")
        return None

    found_files = [initial_kg]

    # Iterate over each relation in the content
    for relation, objects in content["knowledge graph"]["relations"].items():
        for item in objects:
            object_name = item.get("Object")

            # Construct the paths for both original and lowercase filenames
            json_file_original = f"{data_dir}/{object_name}.json"
            json_file_lowercase = f"{data_dir}/{object_name.lower()}.json"

            if storage.file_exists(json_file_original) or storage.file_exists(
                json_file_lowercase
            ):
                json_file = (
                    json_file_original
                    if storage.file_exists(json_file_original)
                    else json_file_lowercase
                )

                try:
                    file_content = storage.load_json(json_file)
                    found_files.append(file_content)
                except Exception as e:
                    print(f"Error loading file {json_file}: {e}")
                    continue

    return found_files
