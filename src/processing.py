import ast
import re

from . import genai
from .entities import Document


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


# get answer
def get_answer(user_question: str, documents: list[Document]) -> str:

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

        DOCS: {documents}
        
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
