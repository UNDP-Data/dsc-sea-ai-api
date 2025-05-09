"""
Routines for database operations for RAG.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import genai, processing, storage


class Client:
    def __init__(self, df, vectoriser):
        self.df = df
        self.vectoriser = vectoriser

    @classmethod
    def from_model(cls) -> "Client":
        df = storage.read_json("models/df_embed_EN_All_V4.jsonl", lines=True)
        # Concatenate 'Document Title' and 'Country Name' to form the text for each document
        df["combined_text"] = (
            df["Document Title"].astype(str) + " " + df["Country Name"].astype(str)
        )
        vectorizer = TfidfVectorizer(stop_words="english")
        df["vector"] = vectorizer.fit_transform(df["combined_text"]).todense().tolist()
        return cls(df=df, vectoriser=vectorizer)

    def filter_semantics(self, user_query: str) -> pd.DataFrame:
        df = self.df.copy()
        # Use TF-IDF Vectorizer to encode the text
        query_embedding = self.vectoriser.transform([user_query])
        # Calculate cosine similarity between the query and document embeddings
        df["similarity_score"] = cosine_similarity(
            query_embedding, np.vstack(df["vector"])
        ).flatten()

        # Filter the DataFrame to include only documents with a similarity score above 0.6
        if not (mask := df["similarity_score"].gt(0.5)).any():
            mask = df["similarity_score"].gt(0.2)
        df = df.loc[mask].copy()
        df.sort_values(
            by="similarity_score",
            ascending=False,
            ignore_index=True,
            inplace=True,
        )
        return df

    def search_embeddings(self, user_query: str) -> pd.DataFrame | None:
        if (df := self.filter_semantics(user_query)).empty:
            return None

        df["similarity_score"] = cosine_similarity(
            [genai.embed_text(user_query)],
            np.vstack(df["Embedding"]),
        ).flatten()
        df.sort_values(
            by="similarity_score",
            ascending=False,
            ignore_index=True,
            inplace=True,
        )
        return df.head()

    @staticmethod
    def map_to_structure(df: pd.DataFrame, user_query: str) -> dict[str, dict]:
        result_dict = {}

        # Create a dictionary for each document
        document_info = {}

        # Counter to limit the loop to 10 iterations
        count = 0
        for index, row in df.iterrows():
            # Define a unique identifier for each document, you can customize this based on your data
            document_id = f"doc-{index + 1}"
            # Handle NaN in content by using fillna
            content = (
                str(row["Content"]) if row["Content"] is not None else ""
            )  # row["Content"]
            content = " ".join(content.split()[:160])

            title = (
                str(row["Document Title"]) if row["Document Title"] is not None else ""
            )
            extract = str(content) if content is not None else ""

            extract_similarity = processing.jaccard_similarity(user_query, extract) or 0

            document_info = {
                "document_title": title,
                "extract": extract,  # Adjust based on your column names
                "category": str(row["Category"]) if row["Category"] is not None else "",
                "document_link": (
                    str(row["Link"]).replace("https-//", "https://")
                    if row["Link"] is not None
                    else ""
                ),
                "summary": (
                    str(row["Summary"]) if row["Summary"] is not None else extract
                ),
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

    def process_queries(self, user_query: str) -> dict[str, dict]:
        merged_result_structure = {}

        # for query in queries:
        df = self.search_embeddings(user_query)
        if df is not None:
            result_structure = self.map_to_structure(df, user_query)
            for doc_id, doc_info in result_structure.items():
                merged_result_structure[doc_id] = doc_info
        return merged_result_structure
