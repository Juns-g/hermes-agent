"""Owned background results continue an ACP conversation after its prompt returns."""

from __future__ import annotations

import asyncio
import logging
import uuid

from acp_adapter.session import SessionState

logger = logging.getLogger("acp_adapter.server")


def _event_identity(event: dict) -> tuple:
    if event.get("type") == "async_delegation":
        failure = ((event.get("results") or [{}])[0] or {}).get("task_index") if event.get("task_failure_notice") else None
        return ("async_delegation", event.get("delegation_id"), failure)
    return ("completion", event.get("session_id"))


class BackgroundNotificationsMixin:
    """Requires the server's connection, session manager and shared turn executor."""

    def _start_background_notifications(self) -> None:
        if self._background_turns_supported and self._background_poller is None:
            self._background_poller = asyncio.create_task(self._background_notification_loop())

    def _enable_background_resume(self, state: SessionState, metadata: dict) -> None:
        hermes = metadata.get("hermes")
        if self._background_turns_supported and isinstance(hermes, dict) and hermes.get("backgroundResume") is True:
            state.background_ready = True

    async def _background_notification_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            try:
                await self._poll_background_notifications_once()
            except Exception:
                logger.exception("ACP background notification poll failed")

    def _owns_background_event(self, state: SessionState, event: dict) -> bool:
        key = str(event.get("session_key") or "")
        if not key:
            return False
        current = str(getattr(state.agent, "session_id", "") or "")
        if key in {state.session_id, current}:
            return True
        db = getattr(state.agent, "_session_db", None)
        if db is None:
            return False
        try:
            return str(db.resolve_resume_session_id(key) or key) == current
        except Exception:
            logger.warning("Could not resolve ACP completion owner", exc_info=True)
            return False

    async def _poll_background_notifications_once(self) -> None:
        if not self._background_turns_supported:
            return
        from tools.process_registry import process_registry

        for state in self.session_manager.loaded_sessions():
            with state.runtime_lock:
                if (not state.background_ready or state.is_running or state.command_op or state.queued_prompts
                        or state.background_cancelled or state.background_task is not None):
                    continue
                notifications = process_registry.drain_notifications(
                    session_key=state.session_id, owns_event=lambda event: self._owns_background_event(state, event),
                    skip_poll_observed=False,
                )
                owned = []
                for event, text in notifications:
                    if event.get("type", "completion") not in {"completion", "async_delegation"} or not self._owns_background_event(state, event):
                        process_registry.completion_queue.put(event)
                    elif _event_identity(event) not in state.background_seen:
                        if self._background_process_notifications_off and event.get("type") != "async_delegation":
                            state.background_seen.add(_event_identity(event))
                        else:
                            owned.append((event, text))
                if not owned:
                    continue
                state.is_running = True
                state.background_task = asyncio.create_task(self._run_background_turn(state, owned))

    async def _run_background_turn(self, state: SessionState, notifications: list) -> None:
        from tools.async_delegation import claim_event_delivery, complete_event_delivery, release_event_delivery
        from tools.process_registry import process_registry
        from tools.process_registry_notifications import ProcessNotificationBatch, group_process_notifications

        claimed = []
        started = False
        accepted = False
        turn_id = str(uuid.uuid4())
        status, error = "failed", None
        try:
            for event, text in notifications:
                if _event_identity(event) in state.background_seen:
                    continue
                try:
                    claim = claim_event_delivery(event, "acp")
                except Exception:
                    logger.exception("Could not claim ACP background result")
                    process_registry.completion_queue.put(event)
                    continue
                if claim is not None:
                    claimed.append((event, text, claim))
            if not claimed or state.background_cancelled:
                return
            texts = []
            for group in group_process_notifications([(event, text) for event, text, _ in claimed]):
                text = ProcessNotificationBatch(group).render(process_registry) if group[0][0].get("type", "completion") == "completion" else group[0][1]
                if text:
                    texts.append(text)
            if not texts:
                return
            await self._conn.ext_notification("hermes/background_turn", {
                "sessionId": state.session_id, "turnId": turn_id, "status": "started",
            })
            started = True
            if state.background_cancelled:
                status = "cancelled"
                return
            accepted = True
            state.background_seen.update(_event_identity(event) for event, _, _ in claimed)
            text = "\n\n".join(texts)
            state.current_prompt_text = text
            response = await self._execute_turn(state, state.session_id, text, text)
            status = "cancelled" if response.stop_reason == "cancelled" else "failed" if state.turn_error else "completed"
            error = state.turn_error
        except Exception as exc:
            error = str(exc)
            logger.exception("ACP background continuation failed for %s", state.session_id)
        finally:
            # Once the model accepted the result, recover its persisted history; never replay its side effects.
            for event, _text, claim in claimed:
                try:
                    if accepted:
                        complete_event_delivery(event, claim)
                    else:
                        release_event_delivery(event, claim)
                        process_registry.completion_queue.put(event)
                except Exception:
                    logger.exception("Could not settle ACP background result delivery")
            if started:
                try:
                    await self._conn.ext_notification("hermes/background_turn", {
                        "sessionId": state.session_id, "turnId": turn_id, "status": status,
                        **({"error": error} if error else {}),
                    })
                except Exception:
                    logger.warning("Could not send ACP background turn end for %s", state.session_id, exc_info=True)
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""
                state.background_task = None

    async def stop_background_notifications(self) -> None:
        if self._background_poller is not None:
            self._background_poller.cancel()
            await asyncio.gather(self._background_poller, return_exceptions=True)
            self._background_poller = None
        tasks = []
        for state in self.session_manager.loaded_sessions():
            if state.background_task is not None:
                tasks.append(state.background_task)
                await self.cancel(state.session_id)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
