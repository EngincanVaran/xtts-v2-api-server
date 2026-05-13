"""
routers/speakers.py — speaker management endpoints.

Endpoints
---------
  GET  /v1/speakers
      List all registered speakers (name + created_at).

  GET  /v1/speakers/{name}
      Metadata for a single speaker.

  DELETE /v1/speakers/{name}
      Remove a speaker from the store and delete its files from disk.

Speaker registration (POST with audio upload) lives in routers/clone.py to
keep the upload/latent-computation logic separate from simple CRUD.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from logging_config import get_logger
from routers.clone import _validate_speaker_name
from speakers import SpeakerNotFoundError, SpeakerStore

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/speakers", tags=["speakers"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SpeakerInfo(BaseModel):
    name: str
    created_at: str


class SpeakerListResponse(BaseModel):
    total: int
    speakers: list[SpeakerInfo]


# ---------------------------------------------------------------------------
# GET /v1/speakers
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=SpeakerListResponse,
    summary="List registered speakers",
    description="Returns name and registration timestamp for every speaker on disk.",
)
async def list_speakers(request: Request) -> SpeakerListResponse:
    store: SpeakerStore = request.app.state.speaker_store
    raw = store.list_speakers()
    speakers = [SpeakerInfo(name=s["name"], created_at=s.get("created_at", "")) for s in raw]
    logger.debug("list_speakers — %d speaker(s)", len(speakers))
    return SpeakerListResponse(total=len(speakers), speakers=speakers)


# ---------------------------------------------------------------------------
# GET /v1/speakers/{name}
# ---------------------------------------------------------------------------

@router.get(
    "/{name}",
    response_model=SpeakerInfo,
    summary="Get speaker metadata",
    description="Returns metadata for a single registered speaker.",
)
async def get_speaker(name: str, request: Request) -> SpeakerInfo:
    _validate_speaker_name(name)  # guard against path traversal via URL segment
    store: SpeakerStore = request.app.state.speaker_store
    try:
        record = store.get(name)
    except SpeakerNotFoundError:
        logger.warning("Speaker not found: %s", name)
        raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found.")
    return SpeakerInfo(name=record.name, created_at=record.created_at)


# ---------------------------------------------------------------------------
# DELETE /v1/speakers/{name}
# ---------------------------------------------------------------------------

class DeleteResponse(BaseModel):
    message: str


@router.delete(
    "/{name}",
    response_model=DeleteResponse,
    summary="Delete a speaker",
    description="Removes the speaker from the store and deletes all files from disk.",
)
async def delete_speaker(name: str, request: Request) -> DeleteResponse:
    _validate_speaker_name(name)  # guard against path traversal via URL segment
    store: SpeakerStore = request.app.state.speaker_store
    try:
        store.delete(name)
    except SpeakerNotFoundError:
        logger.warning("Speaker delete — not found: %s", name)
        raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found.")

    logger.info("Speaker deleted via API: %s", name)
    return DeleteResponse(message=f"Speaker '{name}' deleted.")
