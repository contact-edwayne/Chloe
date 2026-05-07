import time
from hud_server import broadcast_sync

time.sleep(5)
print("Sending: speaking")
broadcast_sync("speaking")
time.sleep(3)
print("Sending: listening")
broadcast_sync("listening")
time.sleep(3)
print("Sending: idle")
broadcast_sync("idle")
print("Done")