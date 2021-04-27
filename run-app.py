import itertools
import threading
import time
import sys

done = False


def animate():
    for c in itertools.cycle(['.', '..', '...']):
        if done:
            break
        sys.stdout.write('\rUpdating stock data' + c)
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write('\rDone!')


t = threading.Thread(target=animate)
t.start()
exec(open('data-clean.py').read())
done = True
print()
time.sleep(2)
exec(open('neural-network.py').read())

