"""
Functions for interacting with GenAI models via Azure OpenAI.
"""

import json
import os
import pkgutil
from typing import AsyncGenerator

import pandas as pd
import yaml
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessageChunk,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from sqlalchemy import StaticPool, create_engine

from .entities import AssistantResponse, Message

__all__ = [
    "get_chat_client",
    "get_embedding_client",
    "generate_response",
    "stream_response",
    "get_sql_tools",
]

PROMPTS = yaml.safe_load(pkgutil.get_data(__name__, "prompts.yaml"))


def get_chat_client(
    temperature: float = 0.0, timeout: int = 10, **kwargs
) -> AzureChatOpenAI:
    """
    Get a chat client for Azure OpenAI service.

    Parameters
    ----------
    temperature : float, default=0.0
        Model temperature setting.
    timeout : int, default=10
        Request timeout setting in seconds.
    **kwargs
        Additional keyword arguments to pass to `AzureChatOpenAI`.

    Returns
    -------
    AzureChatOpenAI
       An Azure OpenAI integration client for chat models.
    """
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_MODEL"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )


def get_embedding_client(**kwargs) -> AzureOpenAIEmbeddings:
    """
    Get an embedding client for Azure OpenAI service.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to `AzureOpenAIEmbeddings`.

    Returns
    -------
    AzureOpenAIEmbeddings
        An Azure OpenAI integration client for embedding.
    """
    return AzureOpenAIEmbeddings(
        model=os.environ["AZURE_OPENAI_EMBED_MODEL"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        # see https://platform.openai.com/docs/api-reference/embeddings#embeddings-create-dimensions
        dimensions=1_024,  # leverage native support for shortening embeddings
        **kwargs,
    )


async def generate_response(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    schema: BaseModel | None = None,
    **kwargs,
) -> str | BaseModel:
    """
    Generate a response using Azure OpenAI service.

    This function supports structured outputs via `response_format` kwarg.

    Parameters
    ----------
    prompt : str
        User message.
    system_message : str, optional
        System message to customise model behaviour.
    schema : BaseModel, optional
        `pydantic` schema for structured output.
    **kwargs
        Addtional keyword arguments to pass to `get_chat_client`.

    Returns
    -------
    str or BaseModel
        String if no `response_format` is specified, otherwise a Pydantic model.
    """
    chat = get_chat_client(**kwargs)
    if schema is not None:
        chat = chat.with_structured_output(schema)
    response = await chat.ainvoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    )
    return response if schema is not None else response.content


async def stream_response(
    messages: MessageLikeRepresentation, tools: list[BaseTool] | None = None, **kwargs
) -> AsyncGenerator[BaseMessageChunk, None]:
    """
    Stream a response from Azure OpenAI service using ReAct Agent.

    Parameters
    ----------
    messages : MessageLikeRepresentation
        Model input as accepted by `astream` method.
    tools : list[BaseTool], optional
        A list of tools the agent can access. If not provided
        the agent will consist of a single LLM node without tools.
    **kwargs
        Addtional keyword arguments to pass to `get_chat_client`.

    Yields
    ------
    BaseMessageChunk
        Messaage chunk from the model.
    """
    chat = get_chat_client(**kwargs)
    agent = create_react_agent(chat, prompt=PROMPTS["answer_question"], tools=tools)
    async for chunk, _ in agent.astream({"messages": messages}, stream_mode="messages"):
        yield chunk


async def extract_entities(user_query: str) -> list[str]:
    """
    Extract relevant entities from the user query.

    Parameters
    ----------
    user_query : str
        Raw user message.

    Returns
    -------
    list[str]
        List of entities extracted from the user message.
    """

    class ResponseFormat(BaseModel):
        """
        Response format for leveraging structured outputs.
        """

        entities: list[str]

    response: ResponseFormat = await generate_response(
        prompt=user_query,
        system_message=PROMPTS["extract_entities"],
        schema=ResponseFormat,
    )
    return response.entities


async def get_answer(
    messages: list[Message],
    response: AssistantResponse,
    tools: list[BaseTool],
) -> AsyncGenerator[str, None]:
    """
    Respond to the user message using RAG and conversation history.

    Parameters
    ----------
    messages : list[Message]
        Conversation history as a list of messages.
    response : AssistantResponse
        Template AssistantResponse to be used to return streamed tokens.
    tools : list[BaseTool]
        List of tool the agent can utilise while generating an answer.

    Yields
    ------
    str
        String representation of JSON model response.
    """
    contents = []
    async for chunk in stream_response(
        messages=[message.to_langchain() for message in messages],
        tools=tools,
        temperature=0.1,
    ):
        if isinstance(chunk, AIMessageChunk):
            # send deltas only
            response.content = chunk.content
            # save the chunks
            contents.append(chunk.content)
            yield response.model_dump_json() + "\n"
            # once the full response has been yielded, nullify properties to reduce payload
            response.clear()
        elif isinstance(chunk, ToolMessage):
            # assign the documents based on tool usage
            if chunk.name == "retrieve_chunks":
                # get the documents from the chunk
                response.documents = chunk.artifact
    else:
        # include the assistant response in the history for generating ideas
        response.documents, response.content = None, ""
        response.ideas = await generate_query_ideas(
            messages + [Message(role="assistant", content="".join(contents))]
        )
        # return the final chunk that includes ideas only
        yield response.model_dump_json() + "\n"


async def generate_query_ideas(messages: list[Message]) -> list[str]:
    """
    Generate query ideas based on the conversation history.

    Parameters
    ----------
    messages : list[Message]
        Conversation history as a list of messages.

    Returns
    -------
    list[str]
        List of query ideas based on the user message.
    """

    class ResponseFormat(BaseModel):
        """
        Response format for leveraging structured outputs.
        """

        ideas: list[str] = Field(
            description="Up to 3 relevant, clear and succint user message ideas."
        )

    response: ResponseFormat = await generate_response(
        prompt=json.dumps([message.model_dump() for message in messages], indent=4),
        system_message=PROMPTS["suggest_ideas"],
        schema=ResponseFormat,
        temperature=0.3,
    )
    return response.ideas


def get_sql_tools(data: list[pd.DataFrame]) -> list[BaseTool]:
    """
    Get SQL tools for an in-memory SQLite database.

    Parameters
    ----------
    data : list[pd.DataFrame]
        List of data frames to be included in the database as table.
        All data frames must contain a `name` property to be used
        as a table name.

    Returns
    -------
    list[BaseTools]
        List of SQL tools for question answering over SQL data.
    """
    engine = create_engine(
        url="sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    # populate the database
    for df in data:
        df.to_sql(df.name, con=engine)
    toolkit = SQLDatabaseToolkit(
        db=SQLDatabase(engine=engine, sample_rows_in_table_info=10),
        llm=get_chat_client(),
    )
    return toolkit.get_tools()
