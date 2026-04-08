"""FastAPI server exposing ProposalCraftEnv endpoints."""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from env.environment import ProposalCraftEnv
from env.models import Action, Observation, StepResult
from env.tasks import TASKS


class ResetRequest(BaseModel):
    """Optional reset body containing a task identifier."""

    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None


app = FastAPI(title="ProposalCraftEnv API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = ProposalCraftEnv()
PORT = int(os.getenv("PORT", "7860"))


@app.post("/reset", response_model=Observation)
def reset_endpoint(payload: ResetRequest | None = None) -> Observation:
    """Reset environment state and optionally select a task."""

    task_id = payload.task_id if payload is not None else None
    return env.reset(task_id=task_id)


@app.post("/step", response_model=StepResult)
def step_endpoint(action: Action) -> StepResult:
    """Apply one environment step from an action payload."""

    return env.step(action)


@app.get("/state")
def state_endpoint() -> dict:
    """Expose current internal state for debugging and monitoring."""

    return env.state()


@app.get("/health")
def health_endpoint() -> dict:
    """Healthcheck endpoint for deployment probes."""

    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def tasks_endpoint() -> list[dict]:
    """List available task metadata."""

    return [
        {
            "id": task.id,
            "name": task.name,
            "difficulty": task.difficulty,
            "max_steps": task.max_steps,
        }
        for task in TASKS.values()
    ]


def main() -> None:
    """Run the API server using uvicorn."""

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
