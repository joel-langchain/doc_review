"""Shared state definition for the regulatory document review graph."""

import operator
from typing import Annotated, TypedDict


class ReviewFinding(TypedDict):
    """A single reviewer's output."""

    reviewer: str
    findings: str


class ReviewState(TypedDict):
    """State passed between nodes in the multi-agent review graph."""

    # The raw regulatory document text being reviewed.
    document: str

    # Each reviewer agent appends its finding here; `operator.add` lets the
    # parallel branches merge their results instead of overwriting one another.
    reviews: Annotated[list[ReviewFinding], operator.add]

    # The final, consolidated report produced by the supervisor agent.
    final_report: str
