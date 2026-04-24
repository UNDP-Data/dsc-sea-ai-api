"""
Functions for interacting with GenAI models via Azure OpenAI.
"""

import asyncio
import json
import logging
import os
import pkgutil
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable

import pandas as pd
import yaml
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from sqlalchemy import StaticPool, create_engine

from .entities import AssistantResponse, Document, Message

__all__ = [
    "get_chat_client",
    "get_embedding_client",
    "generate_response",
    "stream_response",
    "get_sql_tools",
    "assess_scope",
    "build_scope_ideas",
]

PROMPTS = yaml.safe_load(pkgutil.get_data(__name__, "prompts.yaml"))
logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+")
DOMAIN_TERMS = {
    "adaptation",
    "battery",
    "biofuel",
    "carbon",
    "climate",
    "cooking",
    "decarbonization",
    "decarbonisation",
    "electricity",
    "electrification",
    "emissions",
    "energy",
    "esmap",
    "feed",
    "grid",
    "hydropower",
    "interconnection",
    "mini-grid",
    "mitigation",
    "ndc",
    "paris",
    "photovoltaic",
    "renewable",
    "sdg7",
    "sdg",
    "sea",
    "solar",
    "storage",
    "sustainable",
    "tariff",
    "transition",
    "wind",
}
DOMAIN_PHRASES = (
    "access to electricity",
    "clean cooking",
    "climate change",
    "energy access",
    "energy efficiency",
    "energy transition",
    "feed in tariff",
    "feed-in tariff",
    "grid infrastructure",
    "paris agreement",
    "renewable energy",
    "sustainable development",
    "sustainable energy",
    "tracking sdg7",
)
GREETING_PHRASES = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "what can you do",
    "how can you help",
    "help me",
    "who are you",
)
CONVERSATION_META_PHRASES = (
    "remind me",
    "what did i",
    "what did we",
    "earlier in this conversation",
    "in this conversation",
    "previous message",
    "my name",
    "access code",
)
FOLLOW_UP_PHRASES = (
    "tell me more",
    "what about",
    "and what about",
    "how about",
    "can you expand",
    "go deeper",
    "why is that",
)
PROMPT_PROBE_PATTERNS = (
    "system prompt",
    "developer message",
    "hidden instructions",
    "internal instructions",
    "exact instructions",
    "reveal the prompt",
    "show me the prompt",
    "print the prompt",
    "ignore previous instructions",
    "chain of thought",
    "cot",
)
UNSAFE_PATTERNS = (
    "build a bomb",
    "make a bomb",
    "explosive",
    "malware",
    "phishing",
    "ransomware",
    "ddos",
    "steal password",
    "steal credentials",
    "poison",
    "weapon",
)


@dataclass(frozen=True)
class ScopeDecision:
    allowed: bool
    category: str
    reason: str
    refusal: str | None = None


def _normalize_scope_text(text: str | None) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def _contains_domain_signal(text: str | None) -> bool:
    normalized = _normalize_scope_text(text)
    if any(phrase in normalized for phrase in DOMAIN_PHRASES):
        return True
    tokens = set(normalized.split())
    if not tokens:
        return False
    overlap = tokens & DOMAIN_TERMS
    if "feed" in tokens and "tariff" in tokens:
        return True
    if "energy" in tokens:
        return True
    return len(overlap) >= 2


def _is_greeting_or_capability_query(text: str | None) -> bool:
    normalized = _normalize_scope_text(text)
    if not normalized:
        return False
    return any(
        normalized == phrase or normalized.startswith(phrase + " ")
        for phrase in GREETING_PHRASES
    )


def _is_conversation_meta_query(text: str | None) -> bool:
    normalized = _normalize_scope_text(text)
    return any(phrase in normalized for phrase in CONVERSATION_META_PHRASES)


def _is_follow_up_query(text: str | None) -> bool:
    normalized = _normalize_scope_text(text)
    return any(phrase in normalized for phrase in FOLLOW_UP_PHRASES)


def build_scope_ideas(category: str) -> list[str]:
    match category:
        case "prompt_probe":
            return [
                "What can you help me with in sustainable energy?",
                "Explain the connection between renewable energy and climate mitigation.",
                "What is the latest progress on access to electricity?",
            ]
        case "unsafe":
            return [
                "What are the main barriers to expanding renewable energy?",
                "How does climate adaptation differ from mitigation in energy systems?",
                "Tell me more about grid infrastructure for energy transition.",
            ]
        case _:
            return [
                "What is the connection between sustainable energy and climate change mitigation?",
                "Tell me more about access to electricity.",
                "What are the main policy tools for renewable energy deployment?",
            ]


def assess_scope(messages: list[Message]) -> ScopeDecision:
    """
    Apply a deterministic scope guard before any model generation.

    The API is limited to Sustainable Energy Academy topics and should reject
    off-topic, prompt-extraction, and clearly unsafe requests before they reach
    the model runtime.
    """
    if not messages:
        return ScopeDecision(True, "empty", "No messages to assess.")

    enabled = os.getenv("MODEL_SCOPE_GUARD_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return ScopeDecision(True, "disabled", "Scope guard disabled by configuration.")

    latest = messages[-1].content
    latest_normalized = _normalize_scope_text(latest)
    conversation_text = "\n".join(message.content for message in messages[:-1])
    conversation_in_domain = _contains_domain_signal(conversation_text)

    if any(pattern in latest_normalized for pattern in PROMPT_PROBE_PATTERNS):
        return ScopeDecision(
            allowed=False,
            category="prompt_probe",
            reason="Prompt-extraction or instruction-override attempt detected.",
            refusal=(
                "I can't provide system prompts, hidden instructions, or internal configuration. "
                "I can help with Sustainable Energy Academy topics such as sustainable energy, "
                "climate mitigation and adaptation, energy access, grid infrastructure, and SDG 7."
            ),
        )

    if any(pattern in latest_normalized for pattern in UNSAFE_PATTERNS):
        return ScopeDecision(
            allowed=False,
            category="unsafe",
            reason="Clearly unsafe or security-abusive request detected.",
            refusal=(
                "I can't help with harmful, dangerous, or security-abusive requests. "
                "I can help with sustainable energy, climate, energy access, and related SEA topics instead."
            ),
        )

    if _is_greeting_or_capability_query(latest):
        return ScopeDecision(True, "meta", "Greeting or capability query.")

    if _is_conversation_meta_query(latest) and conversation_text.strip():
        return ScopeDecision(True, "conversation_meta", "Conversation memory query.")

    if _contains_domain_signal(latest):
        return ScopeDecision(True, "domain", "Latest query contains in-domain signal.")

    if _is_follow_up_query(latest) and conversation_in_domain:
        return ScopeDecision(True, "follow_up", "Follow-up query grounded in in-domain conversation context.")

    return ScopeDecision(
        allowed=False,
        category="off_topic",
        reason="Latest query is outside the SEA domain scope.",
        refusal=(
            "I’m limited to Sustainable Energy Academy topics such as sustainable energy, "
            "climate mitigation and adaptation, renewable energy, energy access, clean cooking, "
            "grid infrastructure, and related SDG 7 policy questions. Please rephrase your request within that scope."
        ),
    )


def _extract_chunk_text(content: object) -> str:
    """
    Normalize provider chunk content into plain text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                    continue
                if isinstance(text, dict) and isinstance(text.get("value"), str):
                    chunks.append(text["value"])
                    continue
                nested = item.get("content")
                if nested is not None:
                    nested_text = _extract_chunk_text(nested)
                    if nested_text:
                        chunks.append(nested_text)
        return "".join(chunks)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, dict) and isinstance(text.get("value"), str):
            return text["value"]
        nested = content.get("content")
        if nested is not None:
            return _extract_chunk_text(nested)
    return ""


def _extract_tool_documents(chunk: ToolMessage) -> list[Document] | None:
    """
    Recover structured document metadata from a streamed tool message.

    In some LangGraph/LangChain streaming paths, `artifact` may be dropped from
    `ToolMessage` chunks even when the underlying tool returned it. To keep
    document streaming robust, fall back to parsing the tool content payload if
    it includes a `documents` field.
    """
    artifact = getattr(chunk, "artifact", None)
    if isinstance(artifact, list) and artifact:
        documents = []
        for item in artifact:
            if isinstance(item, Document):
                documents.append(item)
            elif isinstance(item, dict):
                try:
                    documents.append(Document.model_validate(item))
                except Exception:
                    continue
        if documents:
            return documents

    raw_content = chunk.content
    if not isinstance(raw_content, str):
        return None
    try:
        payload = json.loads(raw_content)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        return None
    documents = []
    for item in raw_documents:
        if isinstance(item, Document):
            documents.append(item)
        elif isinstance(item, dict):
            try:
                documents.append(Document.model_validate(item))
            except Exception:
                continue
    return documents or None


def _format_conversation(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        speaker = "Assistant" if message.role == "assistant" else "User"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines)


def _build_publications_context(chunks: list[dict]) -> str:
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or "Untitled"
        year = chunk.get("year") or "Unknown year"
        summary = chunk.get("summary") or ""
        content = chunk.get("content") or ""
        content_type = chunk.get("content_type") or ""
        sections.append(
            "\n".join(
                [
                    f"[Publication {index}]",
                    f"Title: {title}",
                    f"Year: {year}",
                    f"Content type: {content_type}",
                    f"Summary: {summary}",
                    f"Excerpt: {content}",
                ]
            )
        )
    return "\n\n".join(sections)


def _trusted_metric_instruction(chunks: list[dict]) -> str:
    """
    Add strict answer requirements for curated, trusted metric fallback chunks.
    """
    for chunk in chunks:
        content = chunk.get("content") or ""
        if (
            chunk.get("content_type") == "trusted_metric_fallback"
            and "666 million" in content
            and "electricity" in content.lower()
        ):
            return (
                "The publication excerpts include a trusted headline metric. The first substantive sentence "
                "of the continuation must state exactly that 666 million people worldwide lacked access to "
                "electricity in 2023. Do not substitute older or approximate global electricity-access figures. "
                "If the user phrased the question as energy access generally, clarify that this figure is for "
                "electricity access; discuss clean cooking only if the supplied excerpts include clean-cooking "
                "evidence."
            )
    return ""


async def stream_chat_response(
    *,
    messages: MessageLikeRepresentation,
    system_message: str,
    **kwargs,
) -> AsyncGenerator[BaseMessageChunk, None]:
    """
    Stream a direct chat completion without tool orchestration.
    """
    chat = get_chat_client(**kwargs)
    prompt_messages = [SystemMessage(content=system_message)]
    for message in list(messages):
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content", "")
            if role == "assistant":
                prompt_messages.append(AIMessage(content=content))
            else:
                prompt_messages.append(HumanMessage(content=content))
        else:
            prompt_messages.append(message)
    async for chunk in chat.astream(prompt_messages):
        yield chunk


def get_chat_client(
    temperature: float = 0.0, timeout: int | None = None, **kwargs
) -> AzureChatOpenAI:
    """
    Get a chat client for Azure OpenAI service.

    Parameters
    ----------
    temperature : float, default=0.0
        Model temperature setting.
    timeout : int | None, default=None
        Request timeout setting in seconds. If not provided, falls back to
        `AZURE_OPENAI_TIMEOUT` env var (default: 60 seconds).
    **kwargs
        Additional keyword arguments to pass to `AzureChatOpenAI`.

    Returns
    -------
    AzureChatOpenAI
       An Azure OpenAI integration client for chat models.
    """
    if timeout is None:
        timeout = int(os.getenv("AZURE_OPENAI_TIMEOUT", "60"))
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
    schema: type[BaseModel] | None = None,
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
        # `json_schema` became the upstream default, but that path leaves parsed
        # Pydantic models attached to message metadata and triggers noisy
        # serializer warnings downstream. `function_calling` preserves the same
        # structured-output contract here without that warning surface.
        chat = chat.with_structured_output(schema, method="function_calling")
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
    publication_task: Awaitable[tuple[list[dict], list[Document]]] | None = None,
    defer_initial_answer: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Respond to the user message using RAG and conversation history.

    Parameters
    ----------
    messages : list[Message]
        Conversation history as a list of messages.
    response : AssistantResponse
        Template AssistantResponse to be used to return streamed tokens.
    publication_task : Awaitable[tuple[list[dict], list[Document]]] | None
        Optional task that resolves to retrieved publication chunks and document metadata.
    defer_initial_answer : bool
        If true, skip the generic draft answer and wait for publication-backed
        evidence first. Intended for current-data queries that should anchor on
        the latest authoritative report.

    Yields
    ------
    str
        String representation of JSON model response.
    """
    heartbeat_raw = os.getenv("MODEL_PUBLICATION_HEARTBEAT_SECONDS", "5")
    try:
        publication_heartbeat_seconds = max(0.05, float(heartbeat_raw))
    except ValueError:
        publication_heartbeat_seconds = 5.0

    def normalize_ideas(raw: object) -> list[str] | None:
        if isinstance(raw, BaseModel):
            raw = getattr(raw, "ideas", None)
        if not isinstance(raw, list):
            return None
        ideas = [idea.strip() for idea in raw if isinstance(idea, str) and idea.strip()]
        return ideas or None

    def reset_response_payload() -> None:
        response.clear()
        response.content = ""

    async def maybe_emit_ideas() -> AsyncGenerator[str, None]:
        nonlocal ideas_payload, ideas_emitted_in_stream
        if ideas_emitted_in_stream:
            return
        if not ideas_task.done():
            return
        try:
            ideas_payload = normalize_ideas(ideas_task.result())
        except Exception as error:
            logger.exception("Error while generating query ideas: %s", error)
            ideas_payload = None
        if ideas_payload:
            response.documents, response.content = None, ""
            response.ideas = ideas_payload
            yield response.model_dump_json() + "\n"
            reset_response_payload()
        ideas_emitted_in_stream = True

    contents: list[str] = []
    ideas_task = asyncio.create_task(generate_query_ideas(messages))
    publication_future: asyncio.Future | asyncio.Task | None = None
    created_publication_task = False
    if publication_task is not None:
        if asyncio.isfuture(publication_task):
            publication_future = publication_task
        else:
            publication_future = asyncio.create_task(publication_task)
            created_publication_task = True
    # Yield once so an already-computable ideas task can start before first token chunks.
    await asyncio.sleep(0)
    ideas_payload: list[str] | None = None
    ideas_emitted_in_stream = False
    try:
        initial_stream_failed = False
        if not defer_initial_answer:
            try:
                async for chunk in stream_chat_response(
                    messages=[message.to_langchain() for message in messages],
                    system_message=PROMPTS["draft_answer"],
                    temperature=0.1,
                ):
                    delta = _extract_chunk_text(getattr(chunk, "content", None))
                    if not delta:
                        continue
                    response.content = delta
                    contents.append(delta)
                    yield response.model_dump_json() + "\n"
                    reset_response_payload()
                    async for ideas_chunk in maybe_emit_ideas():
                        yield ideas_chunk
            except Exception as error:
                logger.exception("Error while streaming initial model response: %s", error)
                initial_stream_failed = True
                reset_response_payload()
                response.content = "I ran into a temporary issue while drafting the initial answer."
                yield response.model_dump_json() + "\n"
                reset_response_payload()
        else:
            response.content = "I will check the publications for the latest data.\n\n"
            yield response.model_dump_json() + "\n"
            reset_response_payload()

        if not defer_initial_answer:
            bridge_text = (
                "\n\nI will check the publications for more insights.\n\n"
                if contents
                else "I will check the publications for more insights.\n\n"
            )
            response.content = bridge_text
            contents.append(bridge_text)
            yield response.model_dump_json() + "\n"
            reset_response_payload()
            async for ideas_chunk in maybe_emit_ideas():
                yield ideas_chunk

        publication_chunks: list[dict] = []
        publication_documents: list[Document] = []
        if publication_future is not None:
            try:
                while not publication_future.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(publication_future),
                            timeout=publication_heartbeat_seconds,
                        )
                    except asyncio.TimeoutError:
                        response.content = ""
                        yield response.model_dump_json() + "\n"
                        reset_response_payload()
                        async for ideas_chunk in maybe_emit_ideas():
                            yield ideas_chunk
                publication_chunks, publication_documents = await publication_future
            except Exception as error:
                logger.exception("Error while retrieving supporting publications: %s", error)
                publication_chunks, publication_documents = [], []

        if publication_documents:
            response.content = ""
            response.documents = publication_documents
            yield response.model_dump_json() + "\n"
            reset_response_payload()
            async for ideas_chunk in maybe_emit_ideas():
                yield ideas_chunk

        if publication_chunks:
            metric_instruction = _trusted_metric_instruction(publication_chunks)
            continuation_prompt = "\n\n".join(
                [
                    f"Conversation history:\n{_format_conversation(messages)}",
                    (
                        "Initial answer already given:\n"
                        + (
                            "No substantive answer was given yet; only a publication lookup notice was sent."
                            if defer_initial_answer
                            else "".join(contents).strip()
                        )
                    ),
                    f"Supporting publication excerpts:\n{_build_publications_context(publication_chunks)}",
                    metric_instruction,
                    (
                        "Continue the answer with additional evidence, examples, figures, or policy insights drawn "
                        "from the publication excerpts. Stay within any country or regional scope named in the "
                        "conversation, and do not generalize from other geographies unless the evidence is clearly "
                        "global and applicable. Do not repeat the initial explanation. Do not output a source list "
                        "or raw URLs."
                    ),
                ]
            )
            try:
                async for chunk in stream_chat_response(
                    messages=[{"role": "user", "content": continuation_prompt}],
                    system_message=PROMPTS["answer_with_publications"],
                    temperature=0.1,
                ):
                    delta = _extract_chunk_text(getattr(chunk, "content", None))
                    if not delta:
                        continue
                    response.content = delta
                    contents.append(delta)
                    yield response.model_dump_json() + "\n"
                    reset_response_payload()
                    async for ideas_chunk in maybe_emit_ideas():
                        yield ideas_chunk
            except Exception as error:
                logger.exception(
                    "Error while streaming publication-grounded continuation: %s",
                    error,
                )
                response.content = (
                    "I found supporting publications, but I ran into a temporary issue while adding the publication-based detail."
                )
                yield response.model_dump_json() + "\n"
                reset_response_payload()
        elif not initial_stream_failed:
            response.content = (
                "I couldn't find closely matching publications to add more detail right now."
            )
            yield response.model_dump_json() + "\n"
            reset_response_payload()
        # include the assistant response in the history for generating ideas
        response.documents, response.content = None, ""
        if ideas_payload is None:
            try:
                ideas_payload = normalize_ideas(await ideas_task)
            except Exception as error:
                logger.exception("Error while generating query ideas: %s", error)
                ideas_payload = None
        response.ideas = ideas_payload
        # return the final chunk that includes ideas only
        yield response.model_dump_json() + "\n"
    finally:
        if publication_future is not None and created_publication_task and not publication_future.done():
            publication_future.cancel()
            try:
                await publication_future
            except asyncio.CancelledError:
                pass
        if not ideas_task.done():
            ideas_task.cancel()
            try:
                await ideas_task
            except asyncio.CancelledError:
                pass


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
