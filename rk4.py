import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mass = 1

def sim(t, y):
    print('called')
    force = np.array([0, 0, -9.81])
    ydot = np.empty((6))
    ydot[0:3] = y[3:6]
    ydot[3:6] = force / mass
    return ydot

y0 = np.array([0, 0, 0, 2, -4, 20])

solution = solve_ivp(sim, (0, 10), y0, max_step=1/10)

print(solution)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(solution.y[0], solution.y[1], solution.y[2])
plt.show()
