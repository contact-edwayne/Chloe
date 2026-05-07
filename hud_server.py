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
                                        "wallet_send", "wallet_history"):
                    if jarvis_handler:
                        await jarvis_handler(data, websocket)
                    else:
                        # Boot race: HUD/PWA WS opens and immediately auto-polls
                        # (e.g. wallet_balance every 30s) before jarvis.py has
                        # finished booting and registered its handler. Silently
                        # drop the request and log to terminal — the next poll
                        # tick will succeed. Previously we replied with
                        # "Jarvis backend not ready yet." which surfaced as a
                        # chat-bubble on every fresh start.
                        print(f"[hud_server] dropping {data.get('type')!r} "
                              f"— jarvis_handler not registered yet", flush=True)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # Plain state string — relay to all other clients
            others = [c for c in hud_clients if c != websocket]
            if others:
                results = await asyncio.gather(
                    *[c.send(message) for c in others],
                    return_exceptions=True
                )
                for r in results:
                    if isinstance(r, Exception):
                        pass  # client disconnected mid-broadcast, ignore
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
            pass  # ignore disconnected clients

async def start_server():
    global server_loop
    server_loop = asyncio.get_event_loop()
    # max_size bumped so mobile WAV blobs (a few seconds of 16kHz mono int16
    # base64 ~ 100-300KB) and TTS audio replies don't bump the websockets
    # default. 8MB gives headroom for longer holds and ElevenLabs MP3 replies.
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
