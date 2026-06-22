# Responsible AI and Governance

## What exists in code

### Source grounding

The `/model` endpoint supports publication-backed answer generation.

Confirmed controls:

- User query is sent to `Client.retrieve_chunks()` to select publication chunks and document references.
- Retrieved chunks are passed into the `answer_with_publications` prompt as explicit context.
- The model is instructed to use the supplied excerpts for factual, statistical, policy, and publication-specific claims.
- Current-data energy-access queries are biased toward the latest Tracking SDG7 report and can use trusted fallback facts when retrieval is incomplete.
- Geographic scope extraction and filtering reject out-of-scope country/region documents when geography is named in the query.
- Moonshot synopsis generation instructs the model to use only supplied dashboard context.

Known limitations:

- The source corpus creation process is not reproducible from raw PDFs in this repository.
- Citation granularity is mostly document-level unless chunk page metadata is available.
- The trusted SDG7 fallback facts are curated code-level snippets, not yet a general fact database.

### Citation and reference handling

Confirmed controls:

- `/model` streams retrieved references as structured `documents[]` chunks.
- The answer body is intentionally kept separate from references.
- Prompts tell the model not to include source lists, inline citations, or raw URLs in answer text.
- `Document.to_stream_payload()` emits only frontend-compatible fields:
  - `title`
  - `year`
  - `language`
  - `url`
  - `summary`
- Document URLs are only emitted when they use `http` or `https`.

Unknown / requires confirmation:

- Whether platform governance treats these references as formal citations.
- Whether page-level citation display is required.
- Whether the frontend preserves every streamed `documents[]` update or replaces the displayed list with the latest payload.

### Prompt controls

Confirmed controls in `src/prompts.yaml` and `src/genai.py`:

- System/developer prompts bind the assistant to the Sustainable Energy Academy and sustainable energy topics.
- The model is instructed not to reveal system prompts, hidden instructions, or internal reasoning.
- The publication continuation prompt instructs the model to stay within requested geographic scope.
- The current-data prompt logic protects key SDG7 figures from being replaced by older approximate values.
- The Moonshot prompts require JSON schema output for query parsing and concise synopsis output for supplied dashboard context.

Known limitations:

- Prompt controls are not sufficient by themselves against all prompt-injection or jailbreak attempts.
- There is no separate model-based prompt-injection classifier in this repository.

### Data restrictions and access controls

Confirmed controls:

- Core API endpoints use `X-Api-Key` authentication.
- Moonshot endpoints use allowed-origin enforcement when `ALLOWED_ORIGINS` is configured.
- Moonshot parse and synopsis endpoints have in-memory rate limits.
- The local KG tester app only allows loopback clients.
- Moonshot Prodoc download-url only allows URLs from the configured Prodoc container prefix and PDF suffix.
- Pydantic request models constrain input lengths on model messages and Moonshot requests.

Known limitations:

- `/model` and graph endpoints do not implement per-user rate limiting in code.
- Authentication is shared API key based, not user-specific RBAC.
- There is no tenant isolation logic in this repository.

### Hallucination mitigation

Confirmed mitigations:

- RAG continuation uses retrieved publication excerpts.
- References are streamed separately so the frontend can expose supporting documents.
- Current-data queries can skip the initial ungrounded draft and check publications first.
- The model is instructed to avoid unsupported source lists and raw URLs.
- The scope guard blocks clearly off-topic, malicious, or prompt-probing requests before model generation.
- Debug retrieval endpoints expose why documents/chunks were selected or rejected.
- Retrieval tests cover regional/national filtering and current SDG7 data behavior.

Known limitations:

- A model may still infer beyond retrieved excerpts unless prompt and evaluation controls catch it.
- Retrieval quality depends on metadata completeness and chunk quality.
- There is no automated factuality checker against source text in production code.

### User data handling

Confirmed behavior:

- `/model` receives conversation messages in request body.
- The service sends message content and generated prompts to Azure OpenAI for response generation.
- Moonshot requests send query/filter/dashboard context to Azure OpenAI or OpenAI, depending on environment configuration.
- `main.py` logs raw query text for `/model` request lifecycle events.
- Request IDs are generated or forwarded with `X-Request-Id` for operational tracing.
- No persistent conversation database is implemented in this repository.

Unknown / requires confirmation:

- Production log retention period.
- Whether logs are centralized in Azure Application Insights or another platform.
- Whether raw user queries are considered personal data under the platform privacy policy.
- Azure OpenAI/OpenAI data retention and processing terms for the deployed tenant.

### Logging of conversations

Confirmed behavior:

- The service logs raw query text in selected operational log lines.
- It logs retrieval timeouts, publication misses, scope blocks, graph timeouts, stream timeouts, and Moonshot provider errors.
- The debug scripts can write raw stream outputs to local files for testing.

Known limitations:

- No explicit conversation-redaction layer exists before logging.
- No structured audit log schema exists for answer provenance, retrieved chunks, model version, and user/session identity.

### Model configuration

Confirmed configuration:

- Primary model path uses Azure OpenAI chat through `AzureChatOpenAI`.
- Embeddings use Azure OpenAI embeddings through `AzureOpenAIEmbeddings`.
- Moonshot can use Azure OpenAI or direct OpenAI.
- Environment variables control model names, endpoint, API version, API keys, and timeout.
- Temperature settings vary by task:
  - deterministic/low-temperature settings for extraction, parsing, and constrained tasks;
  - higher but still bounded settings for some answer generation paths.

Unknown / requires confirmation:

- Exact production model deployments.
- Whether model version upgrades require approval.
- Whether the team maintains a model-change evaluation checklist.

## Missing governance controls requiring confirmation

The following controls are not clearly implemented or documented in this repository:

- Formal data protection impact assessment or privacy review.
- User-facing disclosure that messages are sent to Azure OpenAI/OpenAI.
- PII detection/redaction before logging or model calls.
- Per-user authentication, authorization, quotas, or abuse monitoring for core AI endpoints.
- Semantic safety classifier beyond deterministic scope filtering.
- Prompt-injection classifier for retrieved documents or user messages.
- Source licensing and robots-policy enforcement in ingestion.
- Reproducible raw-document ingestion pipeline from original PDFs/HTML.
- Corpus approval workflow and audit trail.
- Document supersession/withdrawal governance beyond available metadata fields.
- Formal retrieval-quality benchmark gate required before deployment.
- Factuality evaluation against source excerpts.
- Human feedback loop from frontend reference quality or answer quality ratings.
- Page-level citation requirement and verification.
- Conversation retention and deletion policy.
- Secret rotation policy.
- Incident response playbook for hallucination, unsafe output, or data leakage.
