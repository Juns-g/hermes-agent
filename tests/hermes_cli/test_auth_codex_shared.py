"""Opt-in Codex sharing: one owner, one refresh lock, no profile credential copies."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

import pytest

from hermes_cli import auth
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def _jwt(exp):
    payload = base64.urlsafe_b64encode(json.dumps({'exp': exp}).encode()).decode().rstrip('=')
    return f'h.{payload}.s'


@contextmanager
def _scope(home):
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    token = set_hermes_home_override(home)
    secret_token = set_secret_scope({})
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(token)


def _homes(tmp_path, monkeypatch):
    from agent import secret_scope
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    root.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(secret_scope, '_MULTIPLEX_ACTIVE', True)
    token = _jwt(time.time() - 60)
    store = {'version': 1, 'active_provider': 'nous', 'providers': {
        'openai-codex': {'tokens': {'access_token': token, 'refresh_token': 'old-refresh'}},
        'nous': {'access_token': 'owner-only'}}, 'credential_pool': {}}
    (root / 'auth.json').write_text(json.dumps(store))
    homes = []
    for name in ('a', 'b'):
        home = root / 'profiles' / name
        home.mkdir(parents=True)
        (home / 'config.yaml').write_text('auth:\n  codex_use_default: true\n  adopt_external_logins: false\n')
        (home / 'auth.json').write_text(json.dumps({'version': 1, 'providers': {}, 'credential_pool': {}}))
        homes.append(home)
    return root, homes


def test_explicit_codex_sharing_keeps_other_providers_and_scope_isolated(tmp_path, monkeypatch):
    root, homes = _homes(tmp_path, monkeypatch)
    from hermes_cli.auth_codex import codex_auth_file_path
    for home in (homes[0], homes[1], homes[0]):
        with _scope(home):
            assert codex_auth_file_path() == root / 'auth.json'
            assert auth.get_provider_auth_state('nous') is None
            assert auth.read_credential_pool('nous') == []
            runtime = auth.resolve_codex_runtime_credentials(refresh_if_expiring=False)
            assert runtime['api_key'] == auth.get_provider_auth_state('openai-codex')['tokens']['access_token']
    (homes[1] / 'config.yaml').write_text('auth:\n  codex_use_default: false\n  adopt_external_logins: false\n')
    with _scope(homes[1]):
        assert codex_auth_file_path() == homes[1] / 'auth.json'
        assert auth.get_provider_auth_state('openai-codex') is None
        with pytest.raises(auth.AuthError):
            auth.resolve_codex_runtime_credentials(refresh_if_expiring=False)
    with _scope(homes[0]):
        assert codex_auth_file_path() == root / 'auth.json'


def test_shared_pool_and_singleton_serialize_rotation_and_write_only_owner(tmp_path, monkeypatch):
    root, homes = _homes(tmp_path, monkeypatch)
    from agent.credential_pool import load_pool
    from hermes_cli.runtime_provider import resolve_runtime_provider
    before = [(home / 'auth.json').read_bytes() for home in homes]
    with _scope(homes[1]):
        pool = load_pool('openai-codex')
        entry = pool._entries[0]
    calls = []
    fresh = _jwt(time.time() + 7200)

    def refresh(access_token, refresh_token, **kwargs):
        assert getattr(auth._auth_lock_holder_for(root / 'auth.json'), 'depth', 0) > 0
        assert refresh_token == 'old-refresh'
        calls.append(refresh_token)
        return {'access_token': fresh, 'refresh_token': 'new-refresh', 'last_refresh': auth._utc_now_z()}

    monkeypatch.setattr(auth, 'refresh_codex_oauth_pure', refresh)
    barrier = threading.Barrier(2)

    def run(index):
        with _scope(homes[index]):
            barrier.wait(timeout=10)
            if index:
                return pool._refresh_entry(entry, force=False).access_token
            return auth.resolve_codex_runtime_credentials()['api_key']

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(run, (0, 1))) == [fresh, fresh]
    assert len(calls) == 1
    for home, original in zip(homes, before):
        assert (home / 'auth.json').read_bytes() == original
    stored = json.loads((root / 'auth.json').read_text())
    assert stored['providers']['nous'] == {'access_token': 'owner-only'}
    assert stored['active_provider'] == 'nous'
    assert stored['providers']['openai-codex']['tokens']['refresh_token'] == 'new-refresh'
    assert stored['credential_pool']['openai-codex'][0]['refresh_token'] == 'new-refresh'
    for home in (homes[0], homes[1], homes[0]):
        with _scope(home):
            runtime = resolve_runtime_provider(requested='openai-codex', target_model='gpt-5.4-mini')
            assert runtime['api_key'] == fresh
