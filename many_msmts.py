from qutip import *
from math import sqrt

from random import choices

ket0 = basis(2,0)
ket1 = basis(2,1)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


P0 = 0.7
P1 = 0.3

psi = sqrt(P0)*ket0 + sqrt(P1)*ket1


def measure_qubit(meas_basis, qubit):

    basis_vect0 = meas_basis[0].dag()
    basis_vect1 = meas_basis[1].dag()

    p0 = abs(basis_vect0 * qubit)**2
    p1 = abs(basis_vect1 * qubit)**2

    out = choices([meas_basis[0], meas_basis[1]], [p0, p1])

    return out

    



# Assuming measure_qubit is already defined and psi, ket0, ket1 are initialized
n_0, n_1 = 0, 0
# meas_P_0, meas_P_1 = 0,0
nint = 0
results = [0, 0]

fig, ax = plt.subplots()
bars = ax.bar([r'$\uparrow$', r'$\downarrow$'], results)

bar_labels = [ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '', 
                      ha='center', va='bottom') for bar in bars]

def update_histogram(frame):
    global n_0, n_1, nint, bars
    
    out = measure_qubit(meas_basis=[ket0, ket1], qubit=psi)[0]
    
    if out == ket0:
        n_0 += 1
    else:
        n_1 += 1

    results[0] = round(n_0/(n_0 + n_1), 2)
    results[1] = round(n_1/(n_0 + n_1), 2)
    
    for bar, result, label in zip(bars, results, bar_labels):
        bar.set_height(result)

        label.set_text(f'{result}')
        label.set_y(bar.get_height())

    ax.relim()
    ax.autoscale_view()
    
    plt.title(f"Number of measurements: {nint}")
    nint += 1

# Create animation
ani = FuncAnimation(fig, update_histogram, frames=100, repeat=False)
ani.save('7030.mp4', writer='ffmpeg', fps=10)
# plt.show()
# from PIL import Image
# gif = Image.open('5050.gif')
# gif.save('5050_play_once.gif', loop=0)
# print("saved")


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.animation
# import numpy as np


# def init_animation():
#     global line
#     line, = ax.plot(x, np.zeros_like(x))
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1,1)

# def animate(i):
#     line.set_ydata(np.sin(2*np.pi*i / 50)*np.sin(x))
#     return line,

# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.linspace(0, 2*np.pi, 200)

# ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init_animation, frames=50)
# ani.save('test.gif', writer='imagemagick', fps=30)