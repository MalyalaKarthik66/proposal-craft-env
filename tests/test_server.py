"""Integration tests for FastAPI server endpoints."""

import asyncio
from httpx import ASGITransport, AsyncClient

from server.app import app


def test_health_endpoint() -> None:
    async def _call():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/health")

    response = asyncio.run(_call())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_reset_endpoint() -> None:
    async def _call():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/reset", json={"task_id": "easy_abstract"})

    response = asyncio.run(_call())

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "easy_abstract"
    assert "sections_required" in payload


def test_step_endpoint() -> None:
    async def _call():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/reset", json={"task_id": "easy_abstract"})
            return await client.post(
                "/step",
                json={
                    "section_name": "abstract",
                    "content": (
                        "Students use this app with machine learning to optimize study habits "
                        "and improve performance over time. The app helps study planning "
                        "through adaptive recommendations for students. This abstract outlines "
                        "project objectives and expected performance improvements."
                    ),
                    "action_type": "write",
                },
            )

    response = asyncio.run(_call())

    assert response.status_code == 200
    payload = response.json()
    assert "reward" in payload
    assert "value" in payload["reward"]


def test_state_endpoint() -> None:
    async def _call():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/reset", json={"task_id": "easy_abstract"})
            return await client.get("/state")

    response = asyncio.run(_call())

    assert response.status_code == 200
    payload = response.json()
    assert "task_id" in payload
    assert "step_count" in payload
    assert "done" in payload


def test_tasks_endpoint() -> None:
    async def _call():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/tasks")

    response = asyncio.run(_call())

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 3
