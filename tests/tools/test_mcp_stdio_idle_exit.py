"""An idle stdio child exiting before health proof must trigger recovery, not spin."""
import asyncio
import sys
import time

import pytest

from tools import mcp_tool


@pytest.mark.asyncio
async def test_stdio_exit_before_health_proof_reconnects(tmp_path, monkeypatch):
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    script = tmp_path / "exiting_mcp.py"
    pid_file = tmp_path / "child.pid"
    script.write_text('''
import json, os, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
for line in sys.stdin:
    request = json.loads(line)
    if request.get("method") == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "exiting-test", "version": "1"}}
    elif request.get("method") == "tools/list":
        result = {"tools": []}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
    if request["method"] == "tools/list":
        break
''')
    mcp_tool._ensure_mcp_sdk()
    monkeypatch.setattr(mcp_tool, "_DEFAULT_KEEPALIVE_INTERVAL", 0.1)
    server = mcp_tool.MCPServerTask("idle-exit-test")
    server._config = {"command": sys.executable}
    params = StdioServerParameters(command=sys.executable,
                                  args=[str(script), str(pid_file)])
    wait_for_lifecycle = mcp_tool.MCPServerTask._wait_for_lifecycle_event

    async def wait_after_child_exit(self):
        from tests.waiting import async_wait_until
        # Capture this test child's PID without invoking the process-global orphan reaper.
        self._stdio_child_pids = {int(pid_file.read_text())}
        assert await async_wait_until(self._stdio_children_dead, timeout=3)
        return await wait_for_lifecycle(self)

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_wait_for_lifecycle_event", wait_after_child_exit)
    transport = asyncio.create_task(server._serve_transport(stdio_client(params), "stdio", 3))
    try:
        await asyncio.wait_for(server._ready.wait(), 3)
        cpu_start = time.process_time()
        wall_start = time.monotonic()
        # Bound the broken implementation without cancelling SDK context managers.
        stop = asyncio.get_running_loop().call_later(2, server._shutdown_event.set)
        try:
            reason = await asyncio.wait_for(transport, 5)
        finally:
            stop.cancel()
        print(f"idle_exit: reason={reason}, cpu={time.process_time() - cpu_start:.3f}s, "
              f"wall={time.monotonic() - wall_start:.3f}s")
        assert reason == "reconnect"
        assert not server._session_proven
        assert server._stdio_children_dead()
    finally:
        server._shutdown_event.set()
        if not transport.done():
            await asyncio.wait_for(transport, 5)
