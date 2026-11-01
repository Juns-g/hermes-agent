"""A pre-authority GUI client's chat (Herald's ``source='mobile'`` rows) continues under its own id
only through the owner's explicit claim (``session.resume`` + ``adopt_legacy_gui``).

The standalone backend stored any label a GUI client named, so the label alone proves nothing: an
open keeps the read-only history mount and the Bot Chat door still moves an unclaimed one aside. The
claim is this host's native owner ticket naming one exact unrouted row its own login stamped; it binds
the row through the legacy adoption transaction, after which resume, submit and Bot delivery all land
on the original id with its history and label.
"""
import json
from pathlib import Path

import pytest

from tests.gateway import test_peer_dm_bot_chat_headless as _headless_gateway

_headless = _headless_gateway.headless
_local_owner = _headless_gateway._local_owner

SID = '20261022_202222_eb2523'
OWNER_LOGIN = 'auth:v1:["basic","","owner"]'


@pytest.fixture
def headless(_headless):
    # Explicit profile selection compares homes, so bind the fixture before any actor or receipt.
    authority = _headless.authority
    authority.profile_id = str(Path(authority.db.db_path).parent.resolve())
    return _headless


@pytest.fixture(autouse=True)
def host_owner(monkeypatch):
    import gateway.session_worker_reservation as reservation
    monkeypatch.setattr(reservation, 'owner_worker_subject', lambda: 'uid:1000')  # the fixture's native subject


def _herald_chat(db, *, user_id=OWNER_LOGIN, **columns):
    db.create_session(SID, 'mobile', model='original-gui-model')
    db.set_session_title(SID, 'Bot Chat')
    values = {'created_source': 'mobile', 'user_id': user_id,
              'model_config': json.dumps({'follow_profile_config': True}), **columns}
    db._execute_write(lambda conn: conn.execute(
        f"UPDATE sessions SET {', '.join(f'{k}=?' for k in values)} WHERE id=?", (*values.values(), SID)))
    for text in ('role sentinel', 'pwd'):
        db.append_message(SID, 'user', text)
        db.append_message(SID, 'assistant', 'ok ' + text)
    return SID


def _transcript(db):
    return db._read_all('SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id', (SID,))


async def _call(connection, method, **params):
    return await connection.dispatch({'id': 1, 'method': method, 'params': params})


async def _deliver(authority, key):
    from gateway.session_authorities import served_profile_name
    home = Path(authority.db.db_path).parent.resolve()
    response = await _call(_local_owner(authority), 'bot_relay.deliver', id=key * 32,
                           profile=served_profile_name(home), message='research task', session_id=SID)
    assert 'result' in response, response
    return response['result']


@pytest.mark.asyncio
async def test_a_claimed_gui_chat_resumes_submits_and_takes_bot_delivery_under_its_own_id(headless):
    from tests.waiting import async_wait_until
    from hermes_state_local_migration import LEGACY_PREFIX
    db = headless.authority.db
    _herald_chat(db)
    before = _transcript(db)
    saved_model = {key: db.get_session(SID)[key] for key in ('model', 'model_config')}
    owner = _local_owner(headless.authority)

    # Unclaimed: the open stays the read-only history mount.
    assert 'result' in await _call(owner, 'session.resume', session_id=SID)
    refused = await _call(owner, 'prompt.submit', session_id=SID, text='hi', submission_id='s' * 16)
    assert refused['error']['message'] == 'session_read_only'

    claimed = await _call(owner, 'session.resume', session_id=SID, adopt_legacy_gui=True)
    assert claimed['result']['session_id'] == SID, claimed
    again = await _call(_local_owner(headless.authority), 'session.resume', session_id=SID, adopt_legacy_gui=True)
    assert again['result']['session_id'] == SID, 'a repeated claim answers the binding it made'
    assert {key: db.get_session(SID)[key] for key in saved_model} == saved_model

    submitted = await _call(owner, 'prompt.submit', session_id=SID, text='continue', submission_id='t' * 16)
    assert submitted['result']['ref']['session_id'] == SID, submitted
    delivered = await _deliver(headless.authority, 'a')
    assert delivered['session_id'] == SID and delivered['admission_id'], delivered
    await async_wait_until(lambda: len(headless.executed) == 2, timeout=10, interval=.05)
    assert [sid for sid, _, _ in headless.executed] == [SID, SID]
    from tools.bot_live_delivery import read_delivery_result
    home = Path(db.db_path).parent
    await async_wait_until(lambda: (read_delivery_result(home, 'a' * 32) or {}).get('status') == 'settled',
                           timeout=10, interval=.05)
    assert read_delivery_result(home, 'a' * 32)['reply'] == 'pong'
    duplicate = await _deliver(headless.authority, 'a')
    assert duplicate['admission_id'] == delivered['admission_id'] and duplicate['reply'] == 'pong'
    assert len(headless.executed) == 2, 'same-id inspection never executes another turn'

    row = db.get_session(SID)
    assert (row['title'], row['source'], row['created_source']) == ('Bot Chat', 'mobile', 'mobile')
    assert len(db._read_all('SELECT id FROM sessions')) == 1, 'no replacement chat was minted'
    assert _transcript(db)[:len(before)] == before, 'the original messages keep their ids and content'
    # The binding the claim wrote: the local route columns, the native owner, and its legacy record.
    assert (row['chat_id'], row['user_id'], row['chat_type']) == (SID, 'uid:1000', 'dm') and row['session_key']
    binding = db._read_all('SELECT value FROM state_meta WHERE key=?', (LEGACY_PREFIX + SID,))
    assert json.loads(binding[0]['value'])['legacy_session_id'] == SID
    from hermes_state_local import local_receipt
    policy = local_receipt(db, SID)['policy']
    assert json.loads(policy['request_json'])['follow_profile_config'] is True, 'the Bot Chat keeps following its profile'
    assert policy['source'] == 'gui'


@pytest.mark.parametrize('stamp, columns', [
    (None, {}),                                         # unstamped: names no owner
    ('uid:1001', {}),                                   # another OS user's stamp
    (OWNER_LOGIN, {'session_key': 'agent:main:mobile:dm:phone'}),  # a messaging lane's row
    (OWNER_LOGIN, {'profile_name': 'another-profile'}),
    (OWNER_LOGIN, {'created_source': 'cli'}),           # relabelled later, not created by the GUI client
], ids=['unstamped', 'foreign-user', 'routed', 'foreign-profile', 'relabelled'])
@pytest.mark.asyncio
async def test_a_claim_refuses_rows_without_the_owners_gui_provenance(headless, stamp, columns):
    db = headless.authority.db
    _herald_chat(db, user_id=stamp, **columns)
    refused = await _call(_local_owner(headless.authority), 'session.resume', session_id=SID, adopt_legacy_gui=True)
    assert refused['error']['message'] == 'not_found', refused
    assert db.get_session(SID)['chat_id'] is None and db._read_all(
        "SELECT key FROM state_meta WHERE key LIKE ?", ('%' + SID,)) == []


@pytest.mark.asyncio
async def test_only_the_hosts_native_owner_ticket_claims(headless):
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    from gateway.session_controls import AuthorityConnection
    db = headless.authority.db
    _herald_chat(db)
    caps = sorted(_PURPOSE_CAPABILITIES['interactive'])
    authority = headless.authority
    login = AuthorityConnection(authority, object(), {  # the owner signed in to the Webapp: owner, not native
        'user_id': 'owner', 'provider': 'basic', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'capabilities': caps}, operator=True)
    forwarded = AuthorityConnection(authority, object(), {  # a login a standalone backend forwards locally
        'user_id': OWNER_LOGIN, 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'capabilities': caps, 'native_bootstrap': True}, operator=True)
    for connection in (login, forwarded):
        refused = await _call(connection, 'session.resume', session_id=SID, adopt_legacy_gui=True)
        assert refused['error']['message'] == 'permission_denied', refused
    assert db.get_session(SID)['chat_id'] is None


@pytest.mark.asyncio
async def test_a_claim_never_binds_a_chat_another_writer_holds(headless, monkeypatch):
    import time
    from contextlib import contextmanager
    from hermes_cli import active_sessions
    db = headless.authority.db
    _herald_chat(db)
    owner = _local_owner(headless.authority)
    db._execute_write(lambda conn: conn.execute(
        'INSERT INTO session_turn_leases VALUES(?,?,?,?)', (SID, 'herald', time.time(), time.time() + 60)))
    from hermes_state_errors import SessionTurnLeaseLostError
    with pytest.raises(SessionTurnLeaseLostError):  # the binding transaction's idle guard, rolled back
        await _call(owner, 'session.resume', session_id=SID, adopt_legacy_gui=True)
    assert db.get_session(SID)['chat_id'] is None
    db._execute_write(lambda conn: conn.execute('DELETE FROM session_turn_leases'))

    @contextmanager
    def live_process(*_a, **_k):  # the old GUI backend still has the chat open
        yield True
    monkeypatch.setattr(active_sessions, 'active_session_liveness_guard', live_process)
    live = await _call(owner, 'session.resume', session_id=SID, adopt_legacy_gui=True)
    assert live['error']['message'] == 'runtime_coordination_required', live
    assert db.get_session(SID)['chat_id'] is None and db.get_session(SID)['title'] == 'Bot Chat'


@pytest.mark.asyncio
async def test_an_unclaimed_gui_chat_keeps_the_doors_upstream_rule(headless):
    # No claim, no change: the open adopts nothing and the Bot Chat door moves it aside as before.
    db = headless.authority.db
    _herald_chat(db)
    delivered = await _deliver(headless.authority, 'b')
    assert delivered['session_id'] != SID
    assert db.get_session(SID)['title'] == f'Bot Chat (mobile {SID})' and db.get_session(SID)['chat_id'] is None


def test_the_cli_offers_only_named_owner_stamped_gui_chats(tmp_path):
    # ``hermes sessions adopt --gui <id> --dry-run``: read-only, named ids only, the owner's own rule.
    from hermes_cli.sessions_adopt import claimable_gui_sessions
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        _herald_chat(db)
        db.create_session('unstamped', 'mobile')
    finally:
        db.close()
    rows, refused = claimable_gui_sessions(tmp_path / 'state.db', [SID, 'unstamped', 'missing'])
    assert [row['id'] for row in rows] == [SID] and refused == ['unstamped', 'missing']
