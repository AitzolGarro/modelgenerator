"""
File serving endpoints.
Serves generated assets (images, models, exports) from storage.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import get_settings

router = APIRouter(prefix="/files", tags=["files"])
settings = get_settings()

# MIME types for 3D formats
MIME_TYPES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".obj": "text/plain",
    ".mtl": "text/plain",
    ".stl": "model/stl",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@router.get("/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serve a file from storage.
    Path is relative to STORAGE_ROOT.
    """
    # Security: prevent path traversal
    clean_path = Path(file_path)
    if ".." in clean_path.parts:
        raise HTTPException(400, "Invalid path")

    full_path = settings.STORAGE_ROOT / clean_path

    if not full_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")

    if not full_path.is_file():
        raise HTTPException(400, "Not a file")

    # Resolve MIME type
    suffix = full_path.suffix.lower()
    media_type = MIME_TYPES.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(full_path),
        media_type=media_type,
        filename=full_path.name,
    )
