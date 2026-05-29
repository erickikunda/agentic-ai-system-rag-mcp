"""Document upload and ingestion endpoint."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from genai_lab.config.settings import get_settings
from genai_lab.ingestion.chunking import ChunkingProfile
from genai_lab.ingestion.pipeline import run_ingestion

router = APIRouter()

_ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file


@router.post("/ingest")
async def ingest_files(
    files: list[UploadFile] = File(...),
    dry_run: bool = Form(False),
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' — unsupported type. Allowed: .md .txt .pdf",
            )

    settings = get_settings()
    chunking = ChunkingProfile(
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    # Read all file bytes in the async context before handing off to the thread.
    file_payloads: list[tuple[str, bytes]] = []
    for upload in files:
        data = await upload.read()
        if len(data) > _MAX_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' exceeds the 10 MB limit.",
            )
        file_payloads.append((upload.filename or "upload", data))

    def _run() -> dict[str, object]:
        # run_ingestion is synchronous (LlamaIndex IO); execute in a thread pool.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for filename, data in file_payloads:
                (tmp_path / filename).write_bytes(data)
            result = run_ingestion(
                settings=settings,
                input_dir=tmp_path,
                chunking=chunking,
                dry_run=dry_run,
            )
        return {
            "documents_loaded": result.documents_loaded,
            "nodes_created": result.nodes_created,
            "vector_store": result.vector_store,
            "dry_run": result.dry_run,
            "filenames": [name for name, _ in file_payloads],
        }

    return await asyncio.to_thread(_run)
