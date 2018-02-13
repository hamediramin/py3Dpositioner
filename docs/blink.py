from gpiozero import LED, Button
from time import time, sleep
from random import uniform

led = LED(4)
t = (1/40.)/2 #40 Hz
c = 0

start = time()
while (time()-start) < 5:
    led.on()
    sleep(t)
    led.off()
    c += 1
    sleep(t)
end = time()

print(round((end-start),2),c,round(c/(end-start),2))
