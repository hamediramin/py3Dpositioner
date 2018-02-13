import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,2*np.pi,10)
print(t)
for pt in zip(60*np.cos(t), 60*np.sin(t), np.zeros(10)+42.5):
    print(pt)
