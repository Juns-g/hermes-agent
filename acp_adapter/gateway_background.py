"""Project the gateway owner's autonomous turns into Paseo's negotiated ACP lifecycle."""
from __future__ import annotations


class GatewayBackgroundMixin:
    async def _background_turn_start(self, sid, editor_id, aid, payload):
        if (not self._background_turns_supported or not self._conn or not aid
                or payload.get('display_kind') != 'internal_notification'):
            return
        key = (sid, aid)
        if key in self._background_turn_ids or key in self._background_turn_done:
            return
        await self._conn.ext_notification('hermes/background_turn', {
            'sessionId': editor_id, 'turnId': aid, 'status': 'started'})
        self._background_turn_ids[key] = editor_id

    async def _background_turn_end(self, sid, aid, payload):
        key = (sid, aid)
        editor_id = self._background_turn_ids.pop(key, None)
        if editor_id is None:
            return
        self._background_turn_done[key] = None
        if len(self._background_turn_done) > 256:
            self._background_turn_done.pop(next(iter(self._background_turn_done)))
        status = payload.get('outcome', 'completed')
        if status not in {'completed', 'failed', 'cancelled'}:
            status = 'failed'
        event = {'sessionId': editor_id, 'turnId': aid, 'status': status}
        if status == 'failed':
            event['error'] = str(payload.get('error') or 'Gateway background continuation failed')
        await self._conn.ext_notification('hermes/background_turn', event)
