import ratemeter
import time

r = ratemeter.RateMeter()

while True:
    r.notify([1])
    print("rate", r.get_rate())
    time.sleep(0.1)
