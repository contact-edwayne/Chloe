import asyncio
import json
import os
import websockets

hud_clients = set()
jarvis_handler = None
server_loop = None

# Bind host. Default "0.0.0.0" so the mobile PWA over Tailscale can connect.
# Override with CHLOE_WS_HOST=localhost to restrict to the same machine.
WS_HOST = os.environ.get("CHLOE_WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("CHLOE_WS_PORT", "6789"))

async def handler(websocket):
    global jarvis_handler
    hud_clients.add(websocket)
    print(f"Client connected. Total: {len(hud_clients)}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") in ("chat", "volume", "ptt_start", "ptt_stop", "ptt_audio",
                                        "wallet_balance", "wallet_create_invoice",
                                        "wallet_send", "wallet_history",
                                        "lights_state", "lights_action", "lights_discover",
                                        "lights_rename", "lights_preset_apply",
                                        "lights_preset_save", "lights_preset_delete",
                                        "social_drafts_list", "social_draft_now",
                                        "social_draft_edit", "social_draft_approve",
                                        "social_draft_reject"):
                    if jarvis_handler:
                        await jarvis_handler(data, websocket)
                    else:
                        # Boot race: HUD/PWA WS opens and immediately auto-polls
                        # (e.g. wallet_balance every 30s) before jarvis.py has
                        # finished booting and registered its handler. Silently
                        # drop the request and log to terminal -- the next poll
                        # tick will succeed.
                        print(f"[hud_server] dropping {data.get('type')!r} "
                              f"-- jarvis_handler not registered yet", flush=True)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            others = [c for c in hud_clients if c != websocket]
            if others:
                results = await asyncio.gather(
                    *[c.send(message) for c in others],
                    return_exceptions=True
                )
                for r in results:
                    if isinstance(r, Exception):
                        pass
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        hud_clients.discard(websocket)
        print(f"Client disconnected. Total: {len(hud_clients)}")

async def broadcast(message):
    if not hud_clients:
        return
    results = await asyncio.gather(
        *[c.send(message) for c in hud_clients],
        return_exceptions=True
    )
    for r in results:
        if isinstance(r, Exception):
            pass

async def start_server():
    global server_loop
    server_loop = asyncio.get_event_loop()
    # Start the brain graph HTTP server alongside (separate port). Failure
    # to bind is non-fatal — the chat path keeps working. Restored 2026-05-12
    # after a prior file rewrite (when social-media WS handlers were added)
    # accidentally dropped this block; symptom was "localhost refused to
    # connect" on http://localhost:6790/brain-graph.html.
    try:
        import brain_http
        brain_http.start()
    except Exception as e:
        print(f"[hud_server] brain_http start failed: {e}", flush=True)

    async with websockets.serve(handler, WS_HOST, WS_PORT, max_size=8 * 1024 * 1024):
        shown = "localhost" if WS_HOST in ("127.0.0.1", "localhost") else WS_HOST
        print(f"WebSocket server started on ws://{shown}:{WS_PORT} (bind={WS_HOST})")
        await asyncio.Future()

def broadcast_sync(message):
    if server_loop and hud_clients:
        asyncio.run_coroutine_threadsafe(broadcast(message), server_loop)

def set_jarvis_handler(fn):
    global jarvis_handler
    jarvis_handler = fn
