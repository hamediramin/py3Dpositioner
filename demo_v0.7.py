import RPi.GPIO as GPIO
from gpiozero import LED, Button
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import time
import threading

# assume motors start at the top
# closer to GND:  green, red, yellow, blue
izip = zip
xrange = range

# Options
PLOT = True
DEBUG = False
OP = {  # output pins
    'M1_DIR': 21,
    'M1_STEP': 20,
    'M2_DIR': 16,
    'M2_STEP': 12,
    'M3_DIR': 7,
    'M3_STEP': 8,
}
# Black(A+), Green(A-), Red(B+), Blue(B-)

# Parameters
DELAY = 0.002
STEPS_PER_LEG = 2077  # steps per leg
DPS = 1.8  # deg per step
SPR = 200.0  # steps per rev
JK = 127.017  # pitch from centre in mm
CF = 220.0  # arm length in mm
CB = 30.0  # centre offset in mm
AB = 0.0  # tooling offset in mm
TRACK_LENGTH = 400.0  # length of track in mm


# max radis = 63.5 mm = 110*tan(radians(30))

def test_LED():
    led = LED(4)
    t = (1 / 40.) / 2  # 40 Hz
    c = 0

    start = time.time()
    while (time.time() - start) < 5:
        led.on()
        time.sleep(t)
        led.off()
        c += 1
        time.sleep(t)
    end = time.time()

    print(round((end - start), 2), c, round(c / (end - start), 2))

class Stepper:
    def __init__(self, DIR, STEP, delay=DELAY):
        if DEBUG: print('  setting up motor...')
        self.DIR_PIN = DIR
        self.STEP_PIN = STEP
        self.delay = delay
        self.cur_ang = 0.0
        self.cur_h = TRACK_LENGTH  # assume motor is at the top
        self.dps = DPS  # deg per step
        self.spr = SPR  # steps per rev

        GPIO.setup(DIR, GPIO.OUT)
        GPIO.setup(STEP, GPIO.OUT)

        GPIO.output(DIR, 1)  # 1 is down
        self.dir = dir

    def move_steps(self, steps=1, delay=None):
        if delay == None:
            delay = self.delay
        steps = int(steps)
        if DEBUG: print('   adding ', steps * self.dps, ' deg')
        if steps > 0:
            if self.dir != 1:
                GPIO.output(self.DIR_PIN, 1)
                time.sleep(delay)
                self.dir = 1
        elif steps < 0:
            if self.dir != 0:
                GPIO.output(self.DIR_PIN, 0)
                time.sleep(delay)
                self.dir = 0
        for i in range(abs(steps)):
            GPIO.output(self.STEP_PIN, 1)
            time.sleep(delay)
            GPIO.output(self.STEP_PIN, 0)
            time.sleep(delay)
        self.cur_ang += (steps * self.dps) % 360
        self.cur_h += (steps / STEPS_PER_LEG * TRACK_LENGTH)
        if DEBUG: print('   new ang = ', self.cur_ang)

    def incr_ang(self, ang=1, delay=None):
        self.move_steps(ang / 360.0 * self.spr, delay)

    def set_ang(self, ang=None, delay=None):
        if ang == None:
            return
        self.incr_ang(ang - self.cur_ang, delay)

    def move_to_height(self, h):
        print('moving {} steps'.format((self.cur_h - h) / TRACK_LENGTH * STEPS_PER_LEG))
        self.move_steps((self.cur_h - h) / TRACK_LENGTH * STEPS_PER_LEG)


# calculates linear distance up tower for each motor given an xyz point
def get_req_motor_heights(xyz):
    if DEBUG: print('  calculating req motor ang')

    x, y, z = xyz
    H1 = ((((CF) ** 2) - ((((((JK * 0.8660254) - (x + (CB * 0.8660254))) ** 2) + (
    ((JK * -0.5) - (y + (CB * -0.5))) ** 2)) ** 0.5) ** 2)) ** 0.5 + z + AB)
    H2 = ((CF) ** 2 - ((((0 - x) ** 2) + ((JK - (y + CB)) ** 2)))) ** 0.5 + z + AB
    H3 = ((((CF) ** 2) - (
    ((((JK * -0.8660254) - (x - (CB * 0.8660254))) ** 2) + (((JK * -0.5) - (y + (CB * -0.5))) ** 2)))) ** 0.5 + z + AB)

    if DEBUG: print('  req heights = {}, {}, {}'.format(H1, H2, H3))
    return (H1, H2, H3)


def move_to_height(threadname, m, height):
    m.move_to_height(height)
    print(threadname)

def move_to_xyz(m1, m2, m3, xyz):
    H1, H2, H3 = get_req_motor_heights(xyz)
    print('heights = {0:.2f}, {0:.2f}, {0:.2f}'.format(H1, H2, H3))

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


def test_move_to_xyz(xyz_list):
    m1 = Stepper(OP['M1_DIR'], OP['M1_STEP'])
    m2 = Stepper(OP['M2_DIR'], OP['M2_STEP'])
    m3 = Stepper(OP['M3_DIR'], OP['M3_STEP'])

    for [x, y, z] in xyz_list:
        print('moving to ({}, {}, {})...'.format(x, y, z))
        move_to_xyz(m1, m2, m3, (x, y, z))


def run_tests():

    #xyz_list = [[0, 0, 150], [0, 0, 306], [50, 0, 0], [0, 0, 0]]
    xyz_list = [[0, 0, 150], [0, 50, 150], [0, 0, 150]]
    test_move_to_xyz(xyz_list)
#    test_move_to_xyz(0, 0, 400)
#    test_move_to_xyz(50, 0, 150)
#    test_move_to_xyz(0, 0, 150)
#    test_move_to_xyz(0, 50, 150)



def main():
    if DEBUG: print('initializing...')
    GPIO.setmode(GPIO.BCM)  # refer to pins by GPIO # not pin #

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
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    if PLOT: plt.show(block=False)

    for i in range(1, len(x)):
        if PLOT:
            plt.cla()
            ax.plot(x[:i], y[:i], z[:i], label='path')
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_zlim(-10, 10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        xyz = x[i], y[i], z[i]
        if DEBUG: print('x,y,z = {0:.2f}, {0:.2f}, {0:.2f}'.format(*xyz))
        move_to_xyz(m1, m2, m3, xyz)

        # if PLOT: plt.pause(0.0001)
    if PLOT: plt.pause(0.0001)


if __name__ == '__main__':
    run_tests()
    # main()

