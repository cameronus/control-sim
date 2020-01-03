"""
# 1D semi-implicit euler method

t = 0
dt = 1/100

p = 0
v = 0

f = 10
m = 1

while t <= 10:
    print(f'{t}: p={p} v={v}')
    v += (f / m) * dt
    p += v * dt
    t += dt
"""
"""
# 1D verlet method

t = 0
dt = 1/100

p = 0
p_old = 0

f = 10
m = 1

while t <= 10:
    v = abs(p - p_old) / dt # calculated field
    print(f'{t}: p={p} v={v}')
    p_new = (p + (p - p_old) + ((f / m) * dt * dt))
    p_old, p = p, p_new
    t += dt
"""
"""
# 1D velocity verlet method
t = 0
dt = 1/100

p = 0
v = 0
a = 0

m = 1
drag = 0.1

def apply_forces():
    grav_acc = -9.81
    ext_force = 12
    ext_acc = ext_force / m
    drag_force = 0.5 * drag * v * abs(v)
    drag_acc = drag_force / m
    return grav_acc + ext_acc - drag_acc

while t <= 10:
    print(f'{round(t,2)}: p={round(p, 3)} v={round(v, 3)}')
    p += v * dt + a * dt * dt * 0.5
    a_new = apply_forces()
    v += (a + a_new) * dt * 0.5
    a = a_new
    t += dt
"""
"""
# 3D velocity verlet method with only translation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = 0
t_end = 10
dt = 1/100

p = np.zeros((3))
# v = np.zeros((3))
v = np.array([0, 0, 100.])
a = np.zeros((3))

mass = 1
drag = 0.1

p_hist = np.zeros((int(t_end / dt), 3))

def apply_forces():
    grav_acc = np.array([0, 0, -9.81])
    ext_force = np.array([1, -4, 2])
    drag_force = 0.5 * drag * v * np.absolute(v)
    return grav_acc + (ext_force - drag_force) / mass

while t <= t_end:
    p_hist[int(t / dt)] = p
    if float(round(t, 2)).is_integer():
        print(f'{"{:.2f}".format(t)}: p={p} v={v}')
    p += v * dt + a * dt * dt * 0.5
    a_new = apply_forces()
    v += (a + a_new) * dt * 0.5
    a = a_new
    t += dt

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(p_hist[:,0], p_hist[:,1], p_hist[:,2], 'gray')
plt.show()
"""

# 3D velocity verlet method with translation and rotation
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = 0
t_end = 10
dt = 1/100

p = np.zeros((3))
o = np.quaternion(1, 0, 0, 0)

# v = np.zeros((3))
v = np.array([0, 0, 100.])
s = np.quaternion(1, 0, 0, 0)

a = np.zeros((3))
torque = np.quaternion(1, -.2, 0.25, 1)

mass = 1
drag = 0.1

p_hist = np.zeros((int(t_end / dt), 3))
o_hist = np.zeros((int(t_end / dt), 3))

def apply_forces():
    grav_acc = np.array([0, 0, -9.81])
    ext_force = np.array([1, -4, 2])
    drag_force = 0.5 * drag * v * np.absolute(v)
    return grav_acc + (ext_force - drag_force) / mass

while t <= t_end:
    print(quaternion.as_euler_angles(o))
    p_hist[int(t / dt)] = p
    o_hist[int(t / dt)] = quaternion.as_euler_angles(o)
    if float(round(t, 2)).is_integer():
        print(f'{"{:.2f}".format(t)}: p={p} v={v} r={quaternion.as_euler_angles(o)}')
    p += v * dt + a * dt * dt * 0.5
    o = (o * s * dt + torque * dt * dt * 0.5).normalized()
    a_new = apply_forces()
    v += (a + a_new) * dt * 0.5
    s = (s * torque * dt * 0.5).normalized()
    a = a_new
    t += dt

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(p_hist[:,0], p_hist[:,1], p_hist[:,2], 'gray')
p_hist = p_hist[0::30]
o_hist = o_hist[0::30]
ax.quiver(p_hist[:,0], p_hist[:,1], p_hist[:,2], p_hist[:,0] + np.cos(o_hist[:,1]) * np.cos(o_hist[:,0]), p_hist[:,1] + np.sin(o_hist[:,1]) * np.cos(o_hist[:,0]), p_hist[:,2] + np.sin(o_hist[:,0]), length = 8, normalize = True)
plt.show()
