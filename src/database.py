"""
Routines for database operations for RAG.
"""

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import genai, processing, storage

df = storage.read_json("models/df_embed_EN_All_V4.jsonl", lines=True)


def filter_semantics(user_query: str) -> pd.DataFrame:
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


def search_embeddings(
    user_query: str,
) -> tuple[pd.DataFrame, list[float], list[int]] | None:
    filtered_result = filter_semantics(user_query)
    # Check if the result is not None before assigning it to df_filtered
    df_filtered = filtered_result if filtered_result is not None else None

    if (
        df_filtered is not None and not df_filtered.empty
    ):  # Check if DataFrame is not None and not empty
        length = len(df_filtered.head())
        filtered_embeddings_arrays = np.array(list(df_filtered["Embedding"]))
        index = faiss.IndexFlatIP(filtered_embeddings_arrays.shape[1])
        index.add(filtered_embeddings_arrays)

        user_query_embedding = genai.embed_text(user_query)

        k = min(5, length)
        distances, indices = index.search(np.array([user_query_embedding]), k)
        return df_filtered, distances.tolist(), indices.tolist()
    else:
        return None


def map_to_structure(
    qs: tuple[pd.DataFrame, list[float], list[int]], user_query: str
) -> dict[str, dict]:
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

        extract_similarity = processing.jaccard_similarity(user_query, extract) or 0
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


def process_queries(user_query: str) -> dict[str, dict]:
    merged_result_structure = {}

    # for query in queries:
    qs = search_embeddings(user_query)
    if qs is not None:
        result_structure = map_to_structure(qs, user_query)
        for doc_id, doc_info in result_structure.items():
            merged_result_structure[doc_id] = doc_info
    return merged_result_structure
