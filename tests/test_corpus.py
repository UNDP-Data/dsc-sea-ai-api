"""
Tests for corpus metadata enrichment helpers.
"""

from src import corpus


def test_build_source_record_matches_known_domain():
    definition = corpus.match_source("https://www.undp.org/publications/example")
    record = corpus.build_source_record(definition, timestamp="2026-01-01T00:00:00+00:00")
    assert record.source_id == "undp"
    assert record.authority_tier == "trusted"


def test_enrich_chunk_rows_adds_document_and_chunk_identity():
    document = corpus.build_document_record(
        [
            {
                "title": "Grid Modernization Handbook",
                "year": 2024,
                "language": "en",
                "url": "https://example.org/grid-handbook",
                "summary": "Grid modernization and renewable integration guidance.",
                "content": "Grid planning, storage, and transmission modernization.",
            }
        ]
    )
    key = document.url or document.canonical_title
    rows = corpus.enrich_chunk_rows(
        [
            {
                "title": document.canonical_title,
                "year": document.year,
                "language": document.language,
                "url": document.url,
                "summary": document.summary,
                "content": "Grid planning, storage, and transmission modernization.",
            }
        ],
        documents_by_key={key: document},
    )

    assert rows[0]["document_id"] == document.document_id
    assert rows[0]["chunk_id"]
    assert rows[0]["chunk_index"] == 0


def test_document_record_to_api_carries_additive_metadata():
    record = corpus.build_document_record(
        [
            {
                "title": "2025 Tracking SDG7 Report",
                "year": 2025,
                "language": "en",
                "url": "https://trackingsdg7.esmap.org/downloads/report.pdf",
                "summary": "Latest global electricity access progress and indicators.",
                "content": "Electricity access and renewable energy indicators.",
            }
        ]
    )
    api_document = corpus.document_record_to_api(record)

    assert api_document.document_id == record.document_id
    assert api_document.source == record.source_id
    assert api_document.document_type == record.document_type
    assert api_document.publication_date == record.publication_date
