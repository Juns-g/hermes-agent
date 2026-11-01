"""Native Bot delivery over real local-ticket and loopback WebSocket boundaries.

Endpoint discovery is pinned to this test listener. Ticket mint/redeem, SDK delivery, canonical
dispatch, SQLite binding/FIFO and mailbox receipts are real; model execution is the existing
headless fixture's ``pong`` executor, never a provider call or a production gateway.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.test_legacy_gui_claim import SID, _herald_chat, _transcript
from tests.gateway.test_peer_dm_bot_chat_headless import headless as _headless_fixture

headless = _headless_fixture


@pytest.mark.asyncio
async def test_native_sdk_delivers_to_the_claimed_original_id_and_collects_its_receipt(headless, monkeypatch):
    from gateway.control_socket import GatewayControlServer
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import gateway_runtime
    from hermes_cli.gateway_client import GatewayClient, GatewayRPCError, _session_ticket, gateway_ws_target
    from tests.waiting import async_wait_until
    from tools.bot_live_delivery import authority_delivery, read_delivery_result
    from tui_gateway.contracts.registry import CANONICAL_METHODS, METHODS, validate_params
    from websockets.asyncio.client import connect

    authority = headless.authority
    home = Path(authority.db.db_path).parent.resolve()
    authority.profile_id = str(home)
    runner = authority.runner
    # No voice modes in this fixture; keep the real voice gate on its normal off path.
    runner._voice_mode = {}
    runner._voice_key_for_source = lambda source: f'{source.platform.value}:{source.chat_id}'
    runner.session_runtime_descriptor = {'state': 'ready'}
    runner.session_ticket_store = TicketStore(authority.instance_id, frozenset({authority.profile_id}))
    control = GatewayControlServer(home)
    control.ticket_store = runner.session_ticket_store
    api = None
    _herald_chat(authority.db)
    before = _transcript(authority.db)
    saved_model = {key: authority.db.get_session(SID)[key] for key in ('model', 'model_config')}
    try:
        assert await control.start(), 'private fixture control socket must bind'
        api = await start_gateway_api(runner)
        endpoint = SimpleNamespace(profile_id=authority.profile_id, instance_id=authority.instance_id,
                                   api_origin=api.api_origin, control_home=str(home))
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda target, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        token = await asyncio.to_thread(_session_ticket, home, endpoint)
        url, protocols = gateway_ws_target(endpoint, token)
        async with connect(url, subprotocols=protocols, proxy=None) as socket, GatewayClient(socket) as client:
            resumed = await client.rpc('session.resume', session_id=SID)
            assert resumed['session_id'] == SID
            with pytest.raises(GatewayRPCError, match='session_read_only'):
                await client.rpc('prompt.submit', session_id=SID, text='unclaimed', submission_id='s' * 16)
            claimed = await client.rpc('session.resume', session_id=SID, adopt_legacy_gui=True)
            assert claimed['session_id'] == SID
            assert {key: authority.db.get_session(SID)[key] for key in saved_model} == saved_model
            submitted = await client.rpc('prompt.submit', session_id=SID, text='continue', submission_id='t' * 16)
            assert submitted['ref']['session_id'] == SID

        payload = dict(id='c' * 32, profile='default', session_id=SID, message='research task',
                       author={'id': 'bot:default', 'name': 'default', 'is_bot': True})
        assert validate_params(CANONICAL_METHODS['bot_relay.deliver'], payload)[1] is None
        assert validate_params(METHODS['bot_relay.deliver'], payload)[1] is not None
        delivered = await asyncio.to_thread(authority_delivery, home, payload)
        assert delivered['session_id'] == SID and delivered['admission_id']
        await async_wait_until(lambda: (read_delivery_result(home, payload['id']) or {}).get('status') == 'settled',
                               timeout=10, interval=.05)
        repeated = await asyncio.to_thread(authority_delivery, home, payload)
        assert repeated['admission_id'] == delivered['admission_id'] and repeated['reply'] == 'pong'
        with pytest.raises(GatewayRPCError, match='admission_conflict'):
            await asyncio.to_thread(authority_delivery, home, {**payload, 'message': 'changed payload'})
        with pytest.raises(GatewayRPCError, match='profile_mismatch'):
            await asyncio.to_thread(authority_delivery, home, {**payload, 'id': 'd' * 32, 'profile': 'foreign'})
        assert [sid for sid, _, _ in headless.executed] == [SID, SID]
        row = authority.db.get_session(SID)
        assert (row['title'], row['source'], row['created_source']) == ('Bot Chat', 'mobile', 'mobile')
        assert _transcript(authority.db)[:len(before)] == before
        assert len(authority.db._read_all('SELECT id FROM sessions')) == 1
        assert read_delivery_result(home, 'd' * 32) is None
    finally:
        if api is not None:
            await stop_gateway_api(api)
        await control.stop()
