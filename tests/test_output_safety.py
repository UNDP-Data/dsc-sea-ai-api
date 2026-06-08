"""
Regression tests for output encoding of model-controlled content.
"""

import json
from pathlib import Path

from src.entities import AssistantResponse, Document
from src.kg.types import EdgeV2, GraphV2, NodeV2


def test_assistant_response_html_encodes_model_controlled_strings():
    response = AssistantResponse(
        role="assistant",
        content='<img src=x onerror="alert(1)"> ![track](https://evil.example/pixel.png)',
        ideas=['<svg onload="alert(1)"> ![idea](data:image/svg+xml,evil)'],
        documents=[
            Document(
                title="<b>Unsafe title</b>",
                year=2026,
                language="en",
                url="javascript:alert(1)",
                summary='<img src=x onerror="alert(1)">',
            )
        ],
        graph=GraphV2(
            nodes=[
                NodeV2(
                    name="<script>alert(1)</script>",
                    description='<img src=x onerror="alert(1)">',
                    tier="central",
                )
            ],
            edges=[
                EdgeV2(
                    subject="<script>alert(1)</script>",
                    predicate="relates_to",
                    object="Safe",
                    description="<b>unsafe</b>",
                )
            ],
        ),
    )

    payload = json.loads(response.model_dump_json())
    serialized = json.dumps(payload)

    assert payload["content"] == (
        "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; [image omitted: track]"
    )
    assert payload["ideas"] == [
        "&lt;svg onload=&quot;alert(1)&quot;&gt; [image omitted: idea]"
    ]
    assert payload["documents"][0]["title"] == "&lt;b&gt;Unsafe title&lt;/b&gt;"
    assert payload["documents"][0]["summary"] == "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"
    assert payload["documents"][0]["url"] == ""
    assert payload["graph"]["nodes"][0]["name"] == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert payload["graph"]["edges"][0]["description"] == "&lt;b&gt;unsafe&lt;/b&gt;"
    assert "<img" not in serialized
    assert "<script" not in serialized
    assert "![" not in serialized
    assert "data:image" not in serialized
    assert "evil.example" not in serialized
    assert "javascript:" not in serialized


def test_kg_tester_tooltip_uses_text_nodes_instead_of_inner_html():
    for template_path in (
        Path("frontend/templates/kg_tester.html"),
        Path("templates/kg_tester.html"),
    ):
        template = template_path.read_text(encoding="utf-8")
        assert "tooltip.innerHTML" not in template
        assert "tooltip.replaceChildren()" in template
        assert ".textContent = node.description" in template
