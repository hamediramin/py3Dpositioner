import RPi.GPIO as GPIO
import time
from gpiozero import LED, Button
from random import uniform
import matplotlib as mpl # sudo apt-get install python3-matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# sudo apt-get install python3-scipy
import matplotlib.pyplot as plt
import math
from pyaudio import PyAudio # sudo apt-get install python3-pyaudio

izip = zip
xrange = range

# Options
PLOT = True
DEBUG = True
OP = { # output pins
    'ENABLE': 8,
    'M1_CA_1': 4, #motor 1 coil A pin 1
    'M1_CA_2': 17,
    'M1_CB_1': 27,
    'M1_CB_2': 22,
    'M2_CA_1': 5,
    'M2_CA_2': 6,
    'M2_CB_1': 13,
    'M2_CB_2': 19,
    'M3_CA_1': 12,
    'M3_CA_2': 16,
    'M3_CB_1': 20,
    'M3_CB_2': 21,
    }
#Black(A+), Green(A-), Red(B+), Blue(B-) 

# Parameters
DELAY = 0.01
DPS = 1.8 # deg per step
SPR = 200.0 # steps per rev

def test_LED():
    led = LED(4)
    t = (1/40.)/2 #40 Hz
    c = 0

    start = time.time()
    while (time.time()-start) < 5:
        led.on()
        time.sleep(t)
        led.off()
        c += 1
        time.sleep(t)
    end = time.time()

    print(round((end-start),2),c,round(c/(end-start),2))

def test_matplotlib():
    x = np.linspace(0, 2*np.pi, 50)
    y = np.sin(x)
    y2 = y + 0.1 * np.random.normal(size=x.shape)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'k--')
    ax.plot(x, y2, 'ro')

    # set ticks and tick labels
    ax.set_xlim((0, 2*np.pi))
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(['0', '$\pi$', '2$\pi$'])
    ax.set_ylim((-1.5, 1.5))
    ax.set_yticks([-1, 0, 1])

    # Only draw spine between the y-ticks
    ax.spines['left'].set_bounds(-1, 1)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()

def test_pyaudio():
    def sine_tone(frequency, duration, volume=1, sample_rate=22050):
        n_samples = int(sample_rate * duration)
        restframes = n_samples % sample_rate

        p = PyAudio()
        stream = p.open(format=p.get_format_from_width(1), # 8bit
                        channels=1, # mono
                        rate=sample_rate,
                        output=True)
        s = lambda t: volume * math.sin(2 * math.pi * frequency * t / sample_rate)
        samples = (int(s(t) * 0x7f + 0x80) for t in xrange(n_samples))
        for buf in izip(*[samples]*sample_rate): # write several samples at a time
            stream.write(bytes(bytearray(buf)))

        # fill remainder of frameset with silence
        stream.write(b'\x80' * restframes)

        stream.stop_stream()
        stream.close()
        p.terminate()

    sine_tone(
        # see http://www.phy.mtu.edu/~suits/notefreqs.html
        frequency=440.00, # Hz, waves per second A4
        duration=3.21, # seconds to play sound
        volume=.01, # 0..1 how loud it is
        # see http://en.wikipedia.org/wiki/Bit_rate#Audio
        sample_rate=22050 # number of samples per second
    )

def test_3D_matplotlib():
    N = 30
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, N)
    z = np.linspace(-2, 2, N)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    
    #ax.plot(x, y, z, label='parametric curve')

##    plt.ion()
    #plt.xlim([-10,10])
    #plt.ylim([-10,10])
    #plt.zlim([-10,10])
    plt.show(block=False)
    
    for i in range(1,len(x)):
        plt.cla()
        ax.plot(x[:i],y[:i],z[:i], label='path')
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.set_zlim(-10,10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        print(x[i], y[i],z[i])
        #plt.pause(0.0001)
    plt.pause(0.0001)
        


##    # plot in real-time
##    plt.axis([0, 10, 0, 1])
##    plt.ion()
##
##    for i in range(10):
##        y = np.random.random()
##        plt.scatter(i, y)
##        plt.pause(0.05)
##
####    while True:
####        plt.pause(0.05)

def test_move_1_rev():
    print('moving 1 rev...')
    m1 = Stepper(OP['M1_CA_1'], OP['M1_CA_2'], OP['M1_CB_1'], OP['M1_CB_2'])
    m2 = Stepper(OP['M2_CA_1'], OP['M2_CA_2'], OP['M2_CB_1'], OP['M2_CB_2'])
    m3 = Stepper(OP['M3_CA_1'], OP['M3_CA_2'], OP['M3_CB_1'], OP['M3_CB_2'])

    m1.move_steps(1)
    m1.move_steps(-1)
    m2.move_steps(1)
    m2.move_steps(-1)
    m3.move_steps(1)
    m3.move_steps(-1)

    m1.incr_ang(360.0)
    #m1.incr_ang(-360.0)

    time.sleep(1)
    m1.move_steps(200)
    
class Stepper:
    def __init__(self, CA_1, CA_2, CB_1, CB_2, delay=DELAY):
        if DEBUG: print('  setting up motor...')
        self.CA_1_PIN = CA_1
        self.CA_2_PIN = CA_2
        self.CB_1_PIN = CB_1
        self.CB_2_PIN = CB_2
        self.delay = delay
        self.cur_ang = 0.0
        self.dps = DPS # deg per step
        self.spr = SPR # steps per rev

        GPIO.setup(CA_1, GPIO.OUT)
        GPIO.setup(CA_2, GPIO.OUT)
        GPIO.setup(CB_1, GPIO.OUT)
        GPIO.setup(CB_2, GPIO.OUT)

    def move_steps(self, steps=1, delay=None):
        if delay == None:
            delay = self.delay
        steps = int(steps)
        if DEBUG: print('   adding ',steps * self.dps, ' deg')
        if steps > 0:
            for i in range(steps):
                self._set_step(1, 0, 1, 0)
                time.sleep(delay)
                self._set_step(0, 1, 1, 0)
                time.sleep(delay)
                self._set_step(0, 1, 0, 1)
                time.sleep(delay)
                self._set_step(1, 0, 0, 1)
                time.sleep(delay)
        elif steps < 0:
            for i in range(steps):
                self._set_step(1, 0, 0, 1)
                time.sleep(delay)
                self._set_step(0, 1, 0, 1)
                time.sleep(delay)
                self._set_step(0, 1, 1, 0)
                time.sleep(delay)
                self._set_step(1, 0, 1, 0)
                time.sleep(delay)
        self.cur_ang += (steps * self.dps) % 360
        if DEBUG: print('   new ang = ', self.cur_ang)

    def incr_ang(self, ang=1, delay=None):
        self.move_steps(ang/360.0*self.spr, delay)

    def set_ang(self, ang=None, delay=None):
        if ang == None:
            return
        self.incr_ang(ang - self.cur_ang, delay)

    def _set_step(self, v1, v2, v3, v4):
        GPIO.output(self.CA_1_PIN, v1)
        GPIO.output(self.CA_2_PIN, v2)
        GPIO.output(self.CB_1_PIN, v3)
        GPIO.output(self.CB_2_PIN, v4) 

def get_req_motor_ang(m1, m2, m3, xyz):
    if DEBUG: print('  calculating req motor ang')

    ang1, ang2, ang3 = 10, 10, 10
    if DEBUG: print('  req ang = {}, {}, {}'.format(ang1, ang2, ang3))
    return (ang1, ang2, ang3)

def run_tests():
    #test_LED()
    #test_pyaudio()
    #test_matplotlib()
    #test_pyaudio()
    #test_3D_matplotlib()
    test_move_1_rev()
    
def main():
    if DEBUG: print('initializing...')
    GPIO.setmode(GPIO.BCM) # refer to pins by GPIO # not pin #
    GPIO.setup(OP['ENABLE'], GPIO.OUT)
    GPIO.output(OP['ENABLE'], 1)

    m1 = Stepper(OP['M1_CA_1'], OP['M1_CA_2'], OP['M1_CB_1'], OP['M1_CB_2'])
    m2 = Stepper(OP['M2_CA_1'], OP['M2_CA_2'], OP['M2_CB_1'], OP['M2_CB_2'])
    m3 = Stepper(OP['M3_CA_1'], OP['M3_CA_2'], OP['M3_CB_1'], OP['M3_CB_2'])

    # generate path in x, y, z points
    N = 30
    if PLOT: mpl.rcParams['legend.fontsize'] = 10

    if PLOT: fig = plt.figure()
    if PLOT: ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, N)
    z = np.linspace(-2, 2, N)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    
    if PLOT: plt.show(block=False)
    
    for i in range(1,len(x)):
        if PLOT:
            plt.cla()
            ax.plot(x[:i],y[:i],z[:i], label='path')
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.set_zlim(-10,10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        xyz = x[i], y[i], z[i]
        if DEBUG: print('x,y,z = {0:.2f}, {0:.2f}, {0:.2f}'.format(*xyz))
        (ang1, ang2, ang3) = get_req_motor_ang(m1, m2, m3, xyz)
        m1.set_ang(ang1)
        m2.set_ang(ang2)
        m3.set_ang(ang3)
        
        #if PLOT: plt.pause(0.0001)
    if PLOT: plt.pause(0.0001)
    
if __name__ == '__main__':
    #run_tests()
    main()
    
