"""event_bus.py — tiny in-process pub/sub for brain events.

Used by:
  - wiki_watcher.py: emits {type: 'upserted'|'deleted', node_id, ts}
    every time a wiki page changes on disk.
  - brain_http.py /api/brain/ingest: emits {type: 'ingested', node_id,
    similar, ts} when a drop-in ingest commits.
  - brain_http.py /api/brain/events: subscribes per HTTP request and
    streams the queue out as Server-Sent Events to the brain-graph UI.

No external deps. Thread-safe. Each subscriber gets its own bounded
queue; if a subscriber stops draining, oldest events drop on the floor
(buffer default 64). This keeps a stuck UI tab from accumulating
unbounded memory in the backend.

Event shape is loose — callers decide the keys. The bus only guarantees:
  - 'ts' is filled in with time.time() if the caller didn't set it
  - JSON-serializable values reach the consumer in order

Why not stdlib's logging or asyncio: brain_http is a synchronous
ThreadingHTTPServer and wiki_watcher is a polling thread. A trivial
Queue-based fan-out keeps everything in the same paradigm with zero new
deps.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any


_subscribers: list[queue.Queue] = []
_lock = threading.Lock()
_last_event: dict[str, Any] | None = None  # for diagnostics


def subscribe(maxsize: int = 64) -> queue.Queue:
    """Register a subscriber. Returns the Queue to drain. The caller is
    responsible for `unsubscribe()` on shutdown — leaking a queue means
    every publish() does an extra copy forever."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def publish(event: dict[str, Any]) -> None:
    """Fan out `event` to every subscriber. Best-effort — if a
    subscriber's queue is full, drop its oldest event and try again so
    fresh events reach slow consumers eventually."""
    if 'ts' not in event:
        event['ts'] = time.time()
    global _last_event
    _last_event = event
    with _lock:
        targets = list(_subscribers)
    for q in targets:
        try:
            q.put_nowait(event)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(event)
            except (queue.Empty, queue.Full):
                pass


def subscriber_count() -> int:
    with _lock:
        return len(_subscribers)


def last_event() -> dict[str, Any] | None:
    """Most recent event — used by /api/brain/stats so the brain-stats
    pane can show 'watcher heartbeat: N seconds ago' without holding an
    SSE connection open."""
    return _last_event


if __name__ == '__main__':
    # Tiny self-test.
    q = subscribe(maxsize=4)
    publish({'type': 'test', 'i': 1})
    publish({'type': 'test', 'i': 2})
    while not q.empty():
        print(q.get_nowait())
    unsubscribe(q)
    print(f'subscribers: {subscriber_count()}')
