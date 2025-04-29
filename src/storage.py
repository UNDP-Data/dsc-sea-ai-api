"""
Routines for interacting with Azure Blob Storage.
"""

import json
import os

import pandas as pd
from adlfs import AzureBlobFileSystem

__all__ = ["get_blob_client", "file_exists", "load_json", "read_json", "read_csv"]

CONTAINER_NAME = "knowledge-graph"


def _get_credentials() -> dict:
    """
    Get credentials for the Azure Storage account.

    Returns
    -------
    dict
        Dictionary containing the account name and SAS token.
    """
    return {
        "account_name": os.environ["STORAGE_ACCOUNT_NAME"],
        "sas_token": os.environ["STORAGE_SAS_TOKEN"],
    }


def get_blob_client() -> AzureBlobFileSystem:
    """
    Get an adlfs blob file system client.

    Returns
    -------
    AzureBlobFileSystem
        A file system client
    """
    return AzureBlobFileSystem(**_get_credentials())


def file_exists(path: str) -> bool:
    """
    Check if the file exists in the storage container.

    Parameters
    ----------
    path : str
        Path within the container.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
    """
    client = get_blob_client()
    return client.exists(f"{CONTAINER_NAME}/{path}")


def load_json(path: str) -> dict:
    """
    Load a JSON file from Azure Storage.
    """
    client = get_blob_client()
    with client.open(f"{CONTAINER_NAME}/{path}") as file:
        data = json.load(file)
    return data


def read_json(path: str, **kwargs) -> pd.DataFrame:
    """
    Read a JSON file from Azure Storage as a data frame.

    Parameters
    ----------
    path : str
        Path within the blob container.
    **kwargs
        Extra arguments passed to `pd.read_json`.

    Returns
    -------
    pd.DataFrame
        Contents of the JSON file as a data frame.
    """
    df = pd.read_json(
        f"az://{CONTAINER_NAME}/{path}",
        **kwargs,
        storage_options=_get_credentials(),
    )
    return df


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file from Azure Storage as a data frame.

    Parameters
    ----------
    path : str
        Path within the blob container.
    **kwargs
        Extra arguments passed to `pd.read_csv`.

    Returns
    -------
    pd.DataFrame
        Contents of the CSV file as a data frame.
    """
    df = pd.read_csv(
        f"az://{CONTAINER_NAME}/{path}",
        **kwargs,
        storage_options=_get_credentials(),
    )
    return df
