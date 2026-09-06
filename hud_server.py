import asyncio
import json
import os
import websockets

hud_clients = set()
jarvis_handler = None
server_loop = None

# Boot-signal cache. When jarvis.py broadcasts a one-shot signal (boot_start,
# boot_end) that newly-connecting clients also need to see, it should ALSO
# call cache_for_replay() so any HUD/PWA opening later gets the same message
# on connect. This closes the race where the HUD splash attaches its listener
# after the backend has already broadcast (e.g. PWA reloaded mid-boot).
# Kept tiny on purpose — only signals you'd want a late connection to receive.
_replay_messages: list[str] = []

# Bind host. Default "0.0.0.0" so the mobile PWA over Tailscale can connect.
# Override with CHLOE_WS_HOST=localhost to restrict to the same machine.
WS_HOST = os.environ.get("CHLOE_WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("CHLOE_WS_PORT", "6789"))

async def handler(websocket):
    global jarvis_handler
    hud_clients.add(websocket)
    print(f"Client connected. Total: {len(hud_clients)}")
    # Replay any cached one-shot signals so this client sees them even if
    # the broadcast happened before it connected. Boot signals are the main
    # use case — see cache_for_replay() comment.
    if _replay_messages:
        for msg in _replay_messages:
            try:
                await websocket.send(msg)
            except Exception:
                # Client disappeared between accept and replay; the finally
                # block will clean up. Not fatal — silent.
                break
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") in ("chat", "volume", "ptt_start", "ptt_stop", "ptt_audio",
                                        "spotify_control",
                                        "wallet_balance", "wallet_create_invoice",
                                        "wallet_send", "wallet_history",
                                        "lights_state", "lights_action", "lights_discover",
                                        "lights_rename", "lights_preset_apply",
                                        "lights_preset_save", "lights_preset_delete",
                                        "social_drafts_list", "social_draft_now",
                                        "social_draft_edit", "social_draft_approve",
                                        "social_draft_reject",
                                        "jobs_state", "jobs_run",
                                        "logs_tail",
                                        "game_new", "game_move",
                                        "game_state", "game_resign",
                                        "game_watch_start", "game_watch_stop",
                                        "sessions_list", "session_get",
                                        "session_resume", "session_delete",
                                        "session_new"):
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
    clients = list(hud_clients)
    if not clients:
        return

    async def _send_one(c):
        # Bound each send so a half-open client (e.g. a phone that swapped
        # Wi-Fi/cellular over Tailscale and left an orphaned server-side
        # socket) can't stall broadcasts until the ~20s keepalive timeout.
        # Drop the dead client; the PWA's auto-reconnect gives a fresh socket.
        try:
            await asyncio.wait_for(c.send(message), timeout=4)
        except Exception:
            hud_clients.discard(c)
            try:
                asyncio.create_task(c.close())
            except Exception:
                pass

    await asyncio.gather(*[_send_one(c) for c in clients])

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


def cache_for_replay(message):
    """Cache a JSON message string so any client connecting AFTER this call
    receives it on connect. Use for one-shot boot signals that the splash
    screen must see — current callers: _broadcast_boot_start /
    _broadcast_boot_end in jarvis.py.

    Idempotent on dict-vs-str input (accepts either). Safe to call from any
    thread because the list-append is atomic in CPython."""
    if isinstance(message, dict):
        message = json.dumps(message)
    _replay_messages.append(message)


def clear_replay_cache():
    """Forget all cached replay messages. Currently unused — left for future
    callers that may want to invalidate the cache (e.g. on graceful
    shutdown so a reconnecting client doesn't see stale boot signals)."""
    _replay_messages.clear()
