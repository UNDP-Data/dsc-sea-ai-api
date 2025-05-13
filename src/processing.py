import ast
import copy
import re

from . import genai, storage
from .entities import Subgraph


# Extract entities for the query and return the extract entities as an array
def extract_entities(user_query: str) -> list[str]:
    prompt = f"""
    Extract entities from the following user query: \"{user_query}\" and return output in array format.
    
    -Entities should be directly related to the domain or topic of interest. They should represent important concepts that contribute to the understanding of the subject matter.
    -Each entity in the knowledge graph should be distinct and have a unique identifier. This ensures clarity and avoids ambiguity when establishing relationships between entities.
    -You Must return output in array format e.g  ['entity1','entity2'] !!!
    -Avoid adding new lines or breaking spaces to your output. Array should be single dimension and single line !!!
 
    """
    entity_list = genai.generate_response(prompt)
    return ast.literal_eval(entity_list)


# Function to calculate Jaccard similarity between two texts
def jaccard_similarity(text1: str, text2: str) -> float:
    # Tokenize texts
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0


# get answer
def get_answer(user_question: str, relevant_docs: dict[str, dict]) -> str:

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
    system_message = f"""You are a helpful assistant and a professional researcher with many years of experience in answering questions. Give answer to the user's inquiry. {formattings_html}"""
    prompt = f"""{formattings} 
                                        {user_question}
                                        
                                         {formattings_html}
                                        """
    response = genai.generate_response(
        prompt,
        system_message,
        temperature=0.3,
        top_p=0.8,
        frequency_penalty=0.6,
        presence_penalty=0.8,
    )

    # Define the regex pattern to match digits followed by '. do'
    pattern = r"\d+\. do"

    # Remove matches from the text
    cleaned_text = re.sub(pattern, "", response)

    # # Optionally, clean up any extra spaces or punctuation left behind
    # cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()

    return cleaned_text


def remove_thumbnails(data: dict[str, dict]) -> dict[str, dict]:
    data_no_thumbnails = copy.deepcopy(data)  # Make a deep copy of the data
    for doc_id, doc_info in data_no_thumbnails.items():
        if "document_thumbnail" in doc_info:
            del doc_info["document_thumbnail"]

    for doc_id, doc_info in data_no_thumbnails.items():
        if "content" in doc_info:
            del doc_info["content"]

    return data_no_thumbnails


## module to generate query ideas
def generate_query_ideas(user_query: str) -> list[str]:  # lower priority

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
    response = genai.generate_response(prompt)
    query_ideas = response.split(" | ")
    return query_ideas


# Function to calculate the similarity score between two strings
def similarity_score_kg(word1: str, word2: str) -> float:
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


def find_kg(keywords: list[str]) -> list[Subgraph]:
    data_dir = "KG"
    max_score = 0
    most_similar_file = None

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

    return list(map(Subgraph.from_kg, found_files))
