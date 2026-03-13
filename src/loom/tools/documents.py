"""Document retrieval tools — replace stubs with real storage calls."""

from __future__ import annotations


async def download_document(name: str, url: str = "") -> str:
    """Retrieve or download a document by name or URL.

    Args:
        name: Human-readable document name or identifier.
        url: Optional direct URL to the document.

    Returns:
        The document content or a download link.
    """
    # TODO: integrate with your document store (S3, GCS, SharePoint, etc.)
    identifier = url or name
    return (
        f"[STUB] Document '{identifier}' retrieved. "
        "Replace this stub with a real storage/API call."
    )
