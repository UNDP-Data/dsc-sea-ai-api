"""
Shared assertions for SDG7 access metric regressions.
"""

import re

ACCESS_DEFICIT_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(million|billion)\b", re.IGNORECASE)
GLOBAL_MARKERS = ("global", "globally", "worldwide", "world")
ACCESS_MARKERS = (
    "lack",
    "lacked",
    "without access",
    "access to electricity",
    "electricity access",
    "energy access",
)
CLEAN_COOKING_MARKERS = (
    "clean cooking",
    "cooking",
    "cookstove",
    "household air pollution",
    "polluting fuel",
    "polluting fuels",
)
REGIONAL_MARKERS = (
    "sub-saharan africa",
    "africa",
    "asia",
    "latin america",
    "caribbean",
    "east asia",
    "pacific",
    "mena",
    "middle east",
)


def assert_only_666_as_global_electricity_access_deficit(text: str) -> None:
    """
    Ensure answers do not present any non-666 figure as the global electricity-access deficit.

    Separate clean-cooking figures are allowed, and regional electricity-access figures
    are allowed when they are clearly scoped to a region rather than the global total.
    """
    violations = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sentence in sentences:
        lowered = sentence.lower()
        for match in ACCESS_DEFICIT_NUMBER_RE.finditer(sentence):
            phrase = f"{match.group(1)} {match.group(2).lower()}"
            if phrase == "666 million":
                continue
            if any(marker in lowered for marker in CLEAN_COOKING_MARKERS):
                continue
            has_regional_scope = any(marker in lowered for marker in REGIONAL_MARKERS)
            has_global_scope = any(marker in lowered for marker in GLOBAL_MARKERS)
            has_access_context = any(marker in lowered for marker in ACCESS_MARKERS)
            if has_access_context and (has_global_scope or not has_regional_scope):
                violations.append(sentence.strip())
    assert not violations, (
        "Only 666 million may be presented as the global electricity-access deficit; "
        f"found conflicting sentence(s): {violations}"
    )
    assert "666 million" in text
