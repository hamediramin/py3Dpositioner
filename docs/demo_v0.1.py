from gpiozero import LED, Button
from time import time, sleep
from random import uniform
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
from pyaudio import PyAudio # sudo apt-get install python{,3}-pyaudio


izip = zip
xrange = range
        
##led = LED(4)
##t = (1/40.)/2 #40 Hz
##c = 0
##
##start = time()
##while (time()-start) < 5:
##    led.on()
##    sleep(t)
##    led.off()
##    c += 1
##    sleep(t)
##end = time()
##
##print(round((end-start),2),c,round(c/(end-start),2))

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

def main():
    test_pyaudio()

if __name__ == '__main__':
    #test_matplotlib()
    #test_pyaudio()
    test_3D_matplotlib()
    
