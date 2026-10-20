"""Background results resume the owning ACP session without another user prompt."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp.schema import ClientCapabilities, TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState
from tests.waiting import async_wait_until


@pytest.fixture
def harness(monkeypatch, tmp_path):
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    registry._completions_restored = True
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    manager = SessionManager(agent_factory=lambda: None, db=SimpleNamespace())
    manager.save_session = lambda _sid, **_kwargs: None
    conn = SimpleNamespace(session_update=AsyncMock(), ext_notification=AsyncMock(), request_permission=AsyncMock())
    server = HermesACPAgent(session_manager=manager)
    server.on_connect(conn)

    def session(sid):
        model = SimpleNamespace(session_id=sid, calls=[], model="offline")

        def run_conversation(**kwargs):
            model.calls.append(kwargs["user_message"])
            return {"final_response": f"reply-{len(model.calls)}", "messages": []}

        model.run_conversation = run_conversation
        state = SessionState(session_id=sid, agent=model, cwd=str(tmp_path), cancel_event=threading.Event())
        manager._sessions[sid] = state
        return state

    return server, conn, registry, session


async def enable(server):
    return await server.initialize(client_capabilities=ClientCapabilities(field_meta={"hermes": {"backgroundTurns": 1}}))


async def send(server, sid, text="start"):
    return await server.prompt([TextContentBlock(type="text", text=text)], sid)


def completion(sid, process_id="proc-test", code=0):
    return {"type": "completion", "session_key": sid, "session_id": process_id,
            "task_id": sid, "command": "synthetic finite task", "exit_code": code, "output": "result"}


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 7])
async def test_result_automatically_resumes_once_and_emits_lifecycle(harness, code):
    server, conn, registry, session = harness
    state = session("owner")
    try:
        response = await enable(server)
        await send(server, "owner")
        event = completion("owner", code=code)
        registry.completion_queue.put(event)
        assert await async_wait_until(lambda: len(state.agent.calls) == 2, timeout=3)
        assert await async_wait_until(lambda: len(conn.ext_notification.call_args_list) == 2, timeout=3)
        statuses = [call.args[1]["status"] for call in conn.ext_notification.call_args_list]
        assert statuses == ["started", "completed"]
        assert response.agent_capabilities.field_meta["hermes"]["backgroundTurns"] == 1
        assert str(code) in state.agent.calls[1]
        registry.completion_queue.put(event)
        await server._poll_background_notifications_once()
        assert len(state.agent.calls) == 2
    finally:
        await server.stop_background_notifications()


@pytest.mark.asyncio
async def test_foreign_results_and_cancelled_session_are_not_adopted(harness):
    server, _conn, registry, session = harness
    owner = session("owner")
    other = session("other")
    try:
        await enable(server)
        await send(server, "owner")
        await send(server, "other")
        await server.cancel("owner")
        registry.completion_queue.put(completion("owner"))
        registry.completion_queue.put(completion("foreign", "proc-foreign"))
        registry.completion_queue.put(completion("other", "proc-other"))
        assert await async_wait_until(lambda: len(other.agent.calls) == 2, timeout=3)
        assert len(owner.agent.calls) == 1
        assert registry.completion_queue.qsize() == 2
    finally:
        await server.stop_background_notifications()


@pytest.mark.asyncio
async def test_user_prompt_waits_for_background_turn(harness):
    server, _conn, registry, session = harness
    state = session("owner")
    entered = threading.Event()
    release = threading.Event()
    running = 0
    max_running = 0

    def run_conversation(**kwargs):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        text = kwargs["user_message"]
        state.agent.calls.append(text)
        if text != "start" and text != "user follow-up":
            entered.set()
            assert release.wait(5)
        running -= 1
        return {"final_response": text, "messages": []}

    state.agent.run_conversation = run_conversation
    try:
        await enable(server)
        await send(server, "owner")
        registry.completion_queue.put(completion("owner"))
        assert await async_wait_until(entered.is_set, timeout=3)
        user = asyncio.create_task(send(server, "owner", "user follow-up"))
        await asyncio.sleep(0)
        assert not user.done()
        release.set()
        await asyncio.wait_for(user, 5)
        assert max_running == 1
        assert state.agent.calls[-1] == "user follow-up"
    finally:
        release.set()
        await server.stop_background_notifications()


@pytest.mark.asyncio
async def test_plain_acp_client_does_not_receive_vendor_lifecycle(harness):
    server, conn, registry, session = harness
    state = session("owner")
    await server.initialize(client_capabilities=ClientCapabilities())
    await send(server, "owner")
    registry.completion_queue.put(completion("owner"))
    await server._poll_background_notifications_once()
    assert len(state.agent.calls) == 1
    conn.ext_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_rotated_head_claims_only_its_lineage_and_model_failure_is_reported(harness):
    server, conn, registry, session = harness
    state = session("stable")
    try:
        await enable(server)
        await send(server, "stable")
        state.agent.session_id = "head"
        state.agent._session_db = SimpleNamespace(resolve_resume_session_id=lambda key: "head" if key == "old-head" else key)

        def fail(**kwargs):
            state.agent.calls.append(kwargs["user_message"])
            assert kwargs["persist_user_display_kind"] == "hidden"
            raise RuntimeError("model unavailable")

        state.agent.run_conversation = fail
        registry.completion_queue.put(completion("old-head"))
        registry.completion_queue.put(completion("stranger", "proc-foreign"))
        assert await async_wait_until(lambda: len(conn.ext_notification.call_args_list) == 2, timeout=3)
        assert conn.ext_notification.call_args_list[-1].args[1]["status"] == "failed"
        assert registry.completion_queue.qsize() == 1
    finally:
        await server.stop_background_notifications()


@pytest.mark.asyncio
async def test_cancel_during_background_also_cancels_waiting_user_prompt(harness):
    server, conn, registry, session = harness
    state = session("owner")
    entered, release = threading.Event(), threading.Event()

    def run(**kwargs):
        state.agent.calls.append(kwargs["user_message"])
        if len(state.agent.calls) > 1:
            entered.set()
            assert release.wait(5)
        return {"final_response": "result", "messages": []}

    state.agent.run_conversation = run
    try:
        await enable(server)
        await send(server, "owner")
        registry.completion_queue.put(completion("owner"))
        assert await async_wait_until(entered.is_set, timeout=3)
        user = asyncio.create_task(send(server, "owner", "waiting user"))
        await asyncio.sleep(0)
        await server.cancel("owner")
        release.set()
        assert (await asyncio.wait_for(user, 5)).stop_reason == "cancelled"
        assert len(state.agent.calls) == 2
        assert conn.ext_notification.call_args_list[-1].args[1]["status"] == "cancelled"
    finally:
        release.set()
        await server.stop_background_notifications()


@pytest.mark.asyncio
async def test_delivery_failure_before_model_start_keeps_result_for_retry(harness):
    server, conn, registry, session = harness
    state = session("owner")
    try:
        await enable(server)
        await send(server, "owner")
        conn.ext_notification.side_effect = OSError("disconnected")
        registry.completion_queue.put(completion("owner"))
        await server._poll_background_notifications_once()
        task = state.background_task
        assert task is not None
        await task
        assert len(state.agent.calls) == 1
        assert registry.completion_queue.qsize() == 1
        conn.ext_notification.side_effect = None
        await server._poll_background_notifications_once()
        await state.background_task
        assert len(state.agent.calls) == 2
    finally:
        await server.stop_background_notifications()


def test_cancel_latch_survives_restore_without_rewriting_history(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        factory = lambda: SimpleNamespace(model="offline")
        manager = SessionManager(agent_factory=factory, db=db)
        state = manager.create_session(str(tmp_path))
        state.history = [{"role": "user", "content": "preserve me"}, {"role": "assistant", "content": "preserved"}]
        manager.save_session(state.session_id)
        before = db.get_messages_as_conversation(state.session_id)
        state.background_cancelled = True
        state.history = []
        manager.save_session(state.session_id, sync_history=False)
        restored = SessionManager(agent_factory=factory, db=db).get_session(state.session_id)
        assert restored.background_cancelled is True
        assert db.get_messages_as_conversation(state.session_id) == before
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("suppressed", [False, True])
async def test_durable_delegation_is_acknowledged_without_rerunning_child(harness, monkeypatch, suppressed):
    import sqlite3
    from hermes_constants import get_hermes_home
    from tools import async_delegation as delegation

    server, _conn, registry, session = harness
    state = session("owner")
    child_runs = []
    if suppressed:
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"display": {"background_process_notifications": "off"}})

    def child():
        child_runs.append("once")
        return {"results": [{"task_index": 0, "status": "completed", "summary": "child result"}]}

    try:
        await enable(server)
        await send(server, "owner")
        receipt = delegation.dispatch_async_delegation_batch(
            goals=["synthetic child"], context=None, toolsets=None, role="leaf", model="offline",
            session_key="owner", parent_session_id="owner", runner=child,
        )
        assert receipt["status"] == "dispatched"
        assert await async_wait_until(lambda: len(state.agent.calls) == 2, timeout=3)
        assert await async_wait_until(lambda: state.background_task is None, timeout=3)
        with sqlite3.connect(get_hermes_home() / "state.db") as db:
            assert db.execute("SELECT delivery_state FROM async_delegations WHERE delegation_id=?", (receipt["delegation_id"],)).fetchone() == ("delivered",)
        assert delegation.restore_undelivered_completions(registry.completion_queue) == 0
        assert child_runs == ["once"]
    finally:
        await server.stop_background_notifications()


def test_hidden_completion_prompt_is_not_replayed_as_a_user_message():
    from acp_adapter.server import _history_replay_updates

    updates = list(_history_replay_updates([
        {"role": "user", "content": "internal completion", "display_kind": "hidden"},
        {"role": "assistant", "content": "visible result"},
    ]))
    assert len(updates) == 1
    assert updates[0].content.text == "visible result"


def test_only_explicit_interactive_resume_enables_background_delivery(harness):
    server, _conn, _registry, session = harness
    state = session("owner")
    server._background_turns_supported = True
    server._enable_background_resume(state, {"hermes": {"backgroundResume": False}})
    assert state.background_ready is False
    server._enable_background_resume(state, {"hermes": {"backgroundResume": True}})
    assert state.background_ready is True


@pytest.mark.asyncio
async def test_existing_off_setting_suppresses_terminal_wakes(harness, monkeypatch):
    server, conn, registry, session = harness
    state = session("owner")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"display": {"background_process_notifications": "off"}})
    try:
        await enable(server)
        await send(server, "owner")
        registry.completion_queue.put(completion("owner"))
        await server._poll_background_notifications_once()
        assert len(state.agent.calls) == 1
        assert registry.completion_queue.empty()
        conn.ext_notification.assert_not_awaited()
    finally:
        await server.stop_background_notifications()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["claim", "settle"])
async def test_ledger_failure_does_not_leave_session_running(harness, monkeypatch, stage):
    server, _conn, registry, session = harness
    state = session("owner")

    def fail(*_args):
        raise OSError("synthetic ledger failure")

    try:
        await enable(server)
        await send(server, "owner")
        target = "claim_event_delivery" if stage == "claim" else "complete_event_delivery"
        monkeypatch.setattr(f"tools.async_delegation.{target}", fail)
        registry.completion_queue.put(completion("owner"))
        await server._poll_background_notifications_once()
        await state.background_task
        assert state.is_running is False
        assert state.background_task is None
        assert len(state.agent.calls) == (1 if stage == "claim" else 2)
        if stage == "claim":
            assert registry.completion_queue.qsize() == 1
    finally:
        await server.stop_background_notifications()
