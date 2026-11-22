"""Example evaluation dataset for the regulatory document review system.

Each example is a document to review paired with reference notes describing
the issues a good review is expected to surface. These references are used
by the LLM-as-judge evaluators in `evaluators.py`, not for exact matching.
"""

from pathlib import Path

SAMPLE_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sample_documents"


def _load(filename: str) -> str:
    return (SAMPLE_DOCUMENTS_DIR / filename).read_text(encoding="utf-8")


EXAMPLES = [
    {
        "inputs": {"document": _load("sample_clinical_protocol.txt")},
        "outputs": {
            "reference_notes": (
                "A good review should flag: missing Scope section, missing "
                "Risk Assessment section, missing Approval/Sign-off section, "
                "undefined acronyms (PASI, DLQI, IGA, BSA), unsupported "
                "efficacy/safety claims without citations, and incomplete "
                "detail on adverse-event management/long-term safety "
                "monitoring for an immunomodulatory drug."
            )
        },
    },
]


def get_dataset_examples() -> list[dict]:
    """Return the list of {inputs, outputs} example dicts."""
    return EXAMPLES
