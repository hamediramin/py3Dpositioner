RPI = True
try:
    import RPi.GPIO as GPIO
    from gpiozero import LED, Button
except ImportError:
    RPI = False
import time

from random import uniform
import matplotlib as mpl # sudo apt-get install python3-matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# sudo apt-get install python3-scipy
import matplotlib.pyplot as plt
import math
#from pyaudio import PyAudio # sudo apt-get install python3-pyaudio
import threading

# assume motors start at the top
# usable range is z=197 to z=-117 (314 mm)
# usable radius is 100 mm
# closer to GND:  green, red, yellow, blue
izip = zip
xrange = range

# Options
PLOT = False
DEBUG = False
OP = { # output pins
    'M1_DIR': 21,
    'M1_STEP': 20,
    'M2_DIR': 16,
    'M2_STEP': 12,
    'M3_DIR': 7,
    'M3_STEP': 8,
    }
#Black(A+), Green(A-), Red(B+), Blue(B-) 

# Parameters
if RPI:
    DELAY = 0.002
else:
    DELAY = 0
STEPS_PER_LEG = 2077 # steps per leg
DPS = 1.8 # deg per step
SPR = 200.0 # steps per rev
JK = 127.017 # pitch from centre in mm
CF = 220.0 # arm length in mm
CB = 30.0 # centre offset in mm
AB = 0.0 # tooling offset in mm
TRACK_LENGTH = 400.0 # length of track in mm
# max radis = 63.5 mm = 110*tan(radians(30))

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
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])

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

def test_move_constantly():
    print('moving constantly...')
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
##    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
##    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])
    while True:
        m1.move_steps(1)
        time.sleep(0.002)
        
def test_move_to_top():
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])
##    GPIO.output(21, 1) # DIR 1 is up
    for i in range(STEPS_PER_LEG):
##        m1.move_steps(-1)
        m2.move_steps(-1)
##        m3.move_steps(-1)

def test_fast_move():
    d = 0.002
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])
####    GPIO.output(16,0) #DIR=1 is UP
####    time.sleep(d)
####    for i in range(700):
####        GPIO.output(12,1) #STEP
####        time.sleep(d)
####        GPIO.output(12,0) #STEP
####        time.sleep(d)
        
    
    for i in range(200):
        m1.move_steps(-1)
        m2.move_steps(-1)
        m3.move_steps(-1)

##    for i in range(100):
##        m1.move_steps(1)
##        m2.move_steps(1)
##        m3.move_steps(1)

    
class Stepper:
    def __init__(self, DIR, STEP, delay=DELAY):
        if DEBUG: print('  setting up motor...')
        self.DIR_PIN = DIR
        self.STEP_PIN = STEP
        self.delay = delay
        self.cur_ang = 0.0
        self.cur_h = TRACK_LENGTH # assume motor is at the top
        self.dps = DPS # deg per step
        self.spr = SPR # steps per rev
        
        if RPI:
            GPIO.setup(DIR, GPIO.OUT)
            GPIO.setup(STEP, GPIO.OUT)
    
            GPIO.output(DIR, 1) # 1 is down
        self.dir = 1

    def move_steps(self, steps=1, delay=None):
        if delay == None:
            delay = self.delay
        steps = int(steps)
        if DEBUG: print('   adding ',steps * self.dps, ' deg')
        if steps > 0:
            if self.dir != 1:
                if RPI: GPIO.output(self.DIR_PIN, 1)
                time.sleep(delay)
                self.dir = 1
        elif steps < 0:
            if self.dir != 0:
                if RPI: GPIO.output(self.DIR_PIN, 0)
                time.sleep(delay)
                self.dir = 0
        for i in range(abs(steps)):
            if RPI: GPIO.output(self.STEP_PIN, 1)
            time.sleep(delay)
            if RPI: GPIO.output(self.STEP_PIN, 0)
            time.sleep(delay)
        self.cur_ang += (steps*self.dps) % 360
        self.cur_h -= (steps/STEPS_PER_LEG*TRACK_LENGTH)
        if DEBUG: print('   new ang = ', self.cur_ang)

    def incr_ang(self, ang=1, delay=None):
        self.move_steps(ang/360.0*self.spr, delay)

    def set_ang(self, ang=None, delay=None):
        if ang == None:
            return
        self.incr_ang(ang - self.cur_ang, delay)

    def move_to_height(self, h):
        if DEBUG: print('moving {0:.1f} steps'.format((self.cur_h-h)/TRACK_LENGTH*STEPS_PER_LEG))
        self.move_steps((self.cur_h-h)/TRACK_LENGTH*STEPS_PER_LEG)

# calculates linear distance up tower for each motor given an xyz point
def get_req_motor_heights(m1, m2, m3, xyz):
    if DEBUG: print('  calculating req motor ang')

    x, y, z = xyz
    H1 = ((((CF)**2)-((((((JK*0.8660254)-(x+(CB*0.8660254)))**2)+(((JK*-0.5)-(y+(CB*-0.5)))**2))**0.5)**2))**0.5+z+AB)
    H2 = ((CF)**2-((((0-x)**2)+((JK-(y+CB))**2))))**0.5+z+AB
    H3 = ((((CF)**2)-(((((JK*-0.8660254)-(x-(CB*0.8660254)))**2)+(((JK*-0.5)-(y+(CB*-0.5)))**2))))**0.5+z+AB)
    
    if DEBUG: print('  req heights = {}, {}, {}'.format(H1, H2, H3))
    return (H1, H2, H3)

def move_to_height(threadname, m, height):
    m.move_to_height(height)
    #print(threadname)

def move_to_xyz(m1, m2, m3, xyz):
    H1, H2, H3 = get_req_motor_heights(m1, m2, m3, xyz)
    if DEBUG: print('heights = {0:.1f}, {1:.1f}, {2:.1f}'.format(H1, H2, H3))

    try:
        t1 = threading.Thread(name='threadone',
                          target=move_to_height,
                          args=('thread_one', m1, H1))
        t2 = threading.Thread(name='threadtwo',
                          target=move_to_height,
                          args=('thread_two', m2, H2))
        t3 = threading.Thread(name='threadthree',
                          target=move_to_height,
                          args=('thread_three', m3, H3))
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
    except:
        print ("Error: unable to start thread")
        
    m1.move_to_height(H1)
    m2.move_to_height(H2)
    m3.move_to_height(H3)
    
def test_move_to_xyz(x, y, z):
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])

    print('moving to ({0:.2f}, {1:.2f}, {2:.2f})...'.format(x, y, z))
    move_to_xyz(m1, m2, m3, (x, y, z))
    
def test_move_to_xyzs(xyzs):
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])

    for xyz in xyzs:
        x, y, z = xyz
        print('moving to ({0:.2f}, {1:.2f}, {2:.2f})...'.format(x, y, z))
        move_to_xyz(m1, m2, m3, xyz)
        
def test_move_to_height():
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])

    #print('moving to ({0:.2f}, {1:.2f}, {2:.2f})...'.format(x, y, z))
    print('starting at height {0:.1f} mm'.format(m1.cur_h))
    m1.move_to_height(400)
    print('current height {0:.1f} mm'.format(m1.cur_h))
    m1.move_to_height(350)
    print('current height {0:.1f} mm'.format(m1.cur_h))
    m1.move_to_height(370)
    print('current height {0:.1f} mm'.format(m1.cur_h))
    m1.move_to_height(300)
    print('current height {0:.1f} mm'.format(m1.cur_h))
    m1.move_to_height(400)
    print('current height {0:.1f} mm'.format(m1.cur_h))

def test_circle():
    #move in a circle - works!
    N = 100
    t = np.linspace(0,2*np.pi,N)
    z = 202-50
    pts = [[0,0,202],
           [0,0,z]]
    for pt in zip(100*np.cos(t), 100*np.sin(t), np.zeros(N)+z):
        pts.append(pt)
    pts.append([0,0,z])
    pts.append([0,0,202])
    test_move_to_xyzs(pts)

def test_paths():
    #moving from [0,0,202] to [0,0,-197] results in hitting leg 1.
##    test_move_to_xyzs([[0,0,202], #centre of top
##                       [0,0,-197], #centre of bottom
##                       [0,0,0], #centre of middle
##                       [50,0,0], #move in +x direction
##                       [0,0,0],
##                       [-50,0,0], #move in -x direction
##                       [0,0,0],
##                       [0,50,0], #move in +y direction
##                       [0,0,0],
##                       [0,-50,0], #move in -y direction
##                       [0,0,0]])

    #usable range
##    test_move_to_xyzs([[0,0,202], #centre of top
##                       [0,0,-117], #centre of bottom
##                       [0,0,202]
##                       ])

    #works
##    test_move_to_xyzs([[0,0,202], #centre of top
##                       [0,0,-117], #centre of bottom
##                       [0,0,42.5], #middle
##                       [20,0,42.5],
##                       [0,0,42.5],
##                       [0,20,42.5],
##                       [0,0,42.5],
##                       [0,0,202]
##                       ])
    
    test_move_to_xyzs([[0,0,202], #centre of top
                       [0,0,42.5], #middle
                       [20,0,42.5],
                       [0,0,42.5],
                       [30,0,42.5],
                       [0,0,42.5],
                       [40,0,42.5],
                       [0,0,42.5],
                       [50,0,42.5],
                       [0,0,42.5],
                       [60,0,42.5],
                       [0,0,42.5],

                       [0,0,202]
                       ])

def test_spiral():
    #http://mathworld.wolfram.com/ConicalSpiral.html
    if RPI: GPIO.setmode(GPIO.BCM) # refer to pins by GPIO # not pin #

    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])
    
    zhi = 202-10
    zlo = -117+10
    N = 5000
    if PLOT:
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
    theta = np.linspace(-2 * np.pi, 2 * np.pi, N)
    zs = np.linspace(0, zhi-zlo, N)
    xs = zs/3.0 * np.sin(theta)
    ys = zs/3.0 * np.cos(theta)

    zs = zs+zlo
##    ax.plot(x, y, z, label='parametric curve')

##    plt.ion()
    #plt.xlim([-10,10])
    #plt.ylim([-10,10])
    #plt.zlim([-10,10])
    if PLOT: plt.show(block=False)
    
    for i in range(1,len(xs)):
        x, y, z = xs[i-1], ys[i-1], zs[i-1]
        move_to_xyz(m1, m2, m3, (x, y, z))
        if PLOT:
            plt.cla()
            ax.plot(xs[:i],ys[:i],zs[:i], label='path')
            ax.set_xlim(-100,100)
            ax.set_ylim(-100,100)
            ax.set_zlim(-117,202)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.pause(0.0001)
        #print('{0:.1f} {1:.1f} {2:.1f}'.format(x, y, z))
##    plt.pause(0.0001)
    move_to_xyz(m1, m2, m3, (0,0,202))

# DOESN'T WORK YET
def test_helix():
    #http://mathworld.wolfram.com/ConicalSpiral.html
    if RPI: GPIO.setmode(GPIO.BCM) # refer to pins by GPIO # not pin #

##    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
##    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
##    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])
##    
    zhi = 202-10
    zlo = -117+10
    N = 100
    if PLOT:
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
    t = np.linspace(0, 2*np.pi, N)
    xs = np.cos(6*t)
    ys = np.sin(6*t)
    zs = t

    zs = zs+zlo
##    ax.plot(x, y, z, label='parametric curve')

##    plt.ion()
    #plt.xlim([-10,10])
    #plt.ylim([-10,10])
    #plt.zlim([-10,10])
    if PLOT: plt.show(block=False)
    
    for i in range(1,len(xs)):
        x, y, z = xs[i-1], ys[i-1], zs[i-1]
##        move_to_xyz(m1, m2, m3, (x, y, z))
        if PLOT:
            plt.cla()
            ax.plot(xs[:i],ys[:i],zs[:i], label='path')
            ax.set_xlim(-100,100)
            ax.set_ylim(-100,100)
            ax.set_zlim(-117,202)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            #plt.pause(0.0001)
        print('{0:.1f} {1:.1f} {2:.1f}'.format(x, y, z))
    plt.pause(0.0001)
##    move_to_xyz(m1, m2, m3, (0,0,202))
    
def run_tests():
    #test_LED()
    #test_pyaudio()
    #test_matplotlib()
    #test_pyaudio()
    #test_3D_matplotlib()
    #test_move_1_rev()
    #test_move_constantly()
    #test_move_to_top()
    #test_fast_move()
    #test_move_to_height()
    #test_move_to_xyz(0,0,150)
    #test_paths()
    #test_circle()
    test_spiral()
    #test_helix() # DOESN'T WORK YET
    
    #test_move_to_xyz(50,0,150)
    #test_move_to_xyz(0,0,150)
    #test_move_to_xyz(0,50,150)
    
def main():
    if DEBUG: print('initializing...')
    if RPI: GPIO.setmode(GPIO.BCM) # refer to pins by GPIO # not pin #

    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])

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
        if DEBUG: print('x,y,z = {0:.2f}, {1:.2f}, {2:.2f}'.format(*xyz))
        move_to_xyz(m1, m2, m3, xyz)
        
        #if PLOT: plt.pause(0.0001)
    if PLOT: plt.pause(0.0001)
    
if __name__ == '__main__':
    run_tests()
    #main()
    
