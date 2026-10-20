"""Imported historical attachments resolve only into their retained asset roots."""
import json

import pytest

from hermes_cli.web_server_files import _relocated_imported_asset


@pytest.mark.parametrize('owner', ['hermes', 'paseo'])
def test_exact_import_mapping_reads_retained_assets_and_refuses_escape(tmp_path, monkeypatch, owner):
    home = tmp_path / '.hermes'
    roots = {'hermes': home / 'assets' / 'legacy-paths',
             'paseo': tmp_path / '.paseo' / 'attachments' / 'legacy-paths'}
    for root in roots.values():
        root.mkdir(parents=True)
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: home)
    old = tmp_path / 'old-computer' / 'image.png'
    retained = roots[owner] / 'image.png'
    retained.write_bytes(b'original attachment')
    manifest = roots['hermes'] / 'relocations.json'

    def point_at(target):
        manifest.write_text(json.dumps({str(old): str(target)}), encoding='utf-8-sig')

    point_at(retained)
    assert _relocated_imported_asset(old).read_bytes() == b'original attachment'
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'unrelated private file')
    point_at(outside)
    assert _relocated_imported_asset(old) == old
    link = roots[owner] / 'escape.png'
    link.symlink_to(outside)
    point_at(link)
    assert _relocated_imported_asset(old) == old
    old.parent.mkdir()
    old.write_bytes(b'current file')
    point_at(retained)
    assert _relocated_imported_asset(old).read_bytes() == b'current file'
