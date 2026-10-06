"""USER 2026-10-06 (zaragoza): a write to a dead viewer socket must not hang the pipeline —
every write has a timeout, the dead socket is dropped and the broadcast goes on."""

import asyncio
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]


def test_a_dead_viewer_is_dropped_and_the_broadcast_goes_on():
    src = (SERVER / "main.py").read_text()
    i = src.index("class ViewerManager")
    j = src.index("# --- Global State ---")
    ns = {}
    exec("import asyncio\nfrom fastapi import WebSocket\n" + src[i:j], ns)
    VM = ns["ViewerManager"]
    VM._send_timeout_s = staticmethod(lambda n: 0.2)

    class Dead:
        async def send_text(self, m):
            await asyncio.sleep(3600)
        async def close(self):
            pass

    class Live:
        def __init__(self):
            self.got = []
        async def send_text(self, m):
            self.got.append(m)

    async def body():
        vm = VM()
        dead, live = Dead(), Live()
        vm.viewers = {dead, live}
        await asyncio.wait_for(vm.broadcast_text("progress"), timeout=5)
        assert live.got == ["progress"]
        assert dead not in vm.viewers and live in vm.viewers
        await asyncio.wait_for(vm.send_text(None, "again"), timeout=5)
        assert live.got == ["progress", "again"]

    asyncio.run(body())
