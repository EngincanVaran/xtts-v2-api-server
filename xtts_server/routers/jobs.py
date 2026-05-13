"""
routers/jobs.py — job status polling endpoints.

Endpoints
---------
  GET /v1/jobs/{job_id}
      Returns the current status and timing metadata for a job.
      This is the primary polling target after POST /v1/tts or POST /v1/batch.

  GET /v1/jobs
      Lists all jobs currently in the store (useful for dashboards/debugging).
      Not paginated — intended for internal use.

Clients should poll GET /v1/jobs/{job_id} until status is "done" or "failed",
then fetch audio from GET /v1/tts/{job_id}/audio.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from job_store import Job, JobStatus
from logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    job_id: str
    status: str
    language: str
    speaker_id: str
    text_preview: str
    worker_id: str
    gpu_index: int
    # Timing fields — None while the job has not yet reached that stage.
    queue_wait_ms: float | None
    synthesis_ms: float | None
    total_ms: float | None
    # Set when done.
    audio_url: str | None
    # Set when failed.
    error: str | None


def _job_to_response(job: Job) -> JobResponse:
    audio_url = (
        f"/v1/tts/{job.job_id}/audio"
        if job.status == JobStatus.DONE
        else None
    )
    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        language=job.language,
        speaker_id=job.speaker_id,
        text_preview=job.text_preview,
        worker_id=job.worker_id,
        gpu_index=job.gpu_index,
        queue_wait_ms=job.queue_wait_ms(),
        synthesis_ms=job.synthesis_ms(),
        total_ms=job.total_ms(),
        audio_url=audio_url,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Poll job status",
    description=(
        "Returns current status and timing metadata for the given job. "
        "Poll until status is 'done' or 'failed', then fetch audio from "
        "the audio_url field."
    ),
)
async def get_job(job_id: str, request: Request) -> JobResponse:
    job = await request.app.state.job_store.get(job_id)

    if job is None:
        logger.warning("Job not found: %s", job_id)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return _job_to_response(job)


# ---------------------------------------------------------------------------
# GET /v1/jobs
# ---------------------------------------------------------------------------

class JobListResponse(BaseModel):
    total: int
    jobs: list[JobResponse]


@router.get(
    "",
    response_model=JobListResponse,
    summary="List all jobs",
    description="Returns all jobs currently in the in-memory store. For debugging and dashboards.",
)
async def list_jobs(request: Request) -> JobListResponse:
    jobs = await request.app.state.job_store.list_all()
    responses = [_job_to_response(j) for j in jobs]
    return JobListResponse(total=len(responses), jobs=responses)
