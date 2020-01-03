import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *

mass = 1.
inertia_mat = np.matrix([[mass / 6, 0, 0], [0, mass / 6, 0], [0, 0, mass / 6]])

position = np.zeros((3))
# velocity = np.zeros((3))
velocity = np.array([10., 5., 30.])

orientation = np.quaternion(1, 0, 0, 0)
# omega = np.zeros((3))
omega = np.array([1., 0., -0.2])

def apply_forces():
    gravity = np.array([0., 0., -9.81])
    push = np.array([0.2, 0.1, 0.])
    force = gravity + push
    torque = np.cross(push, np.array([0, 0, -0.5]))
    return force, torque

t = 0
t_end = 10
dt = 1/100

p = np.zeros((int(t_end / dt), 3))
o = np.zeros((int(t_end / dt), 3))

while t <= t_end:
    p[int(t / dt)] = position
    o[int(t / dt)] = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
    force, torque = apply_forces()
    velocity += force / mass * dt
    position += velocity * dt
    orientation += orientation * np.quaternion(0, *omega) * 0.5 * dt
    rot_matrix = quaternion.as_rotation_matrix(orientation)
    rot_inertia_mat = np.asarray(rot_matrix * inertia_mat * np.transpose(rot_matrix))
    omega += np.matmul(np.linalg.inv(rot_inertia_mat), (torque - np.cross(omega, np.matmul(rot_inertia_mat, omega)))) * dt
    t += dt

"""
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-400, 400)
ax.set_ylim3d(-400, 400)
ax.set_zlim3d(-400, 400)
# ax.plot3D(p_hist[:,0], p_hist[:,1], p_hist[:,2], 'gray')
ax.plot3D(p[:,0], p[:,1], p[:,2], 'gray')
p = p[::30]
o = o[::30]
ax.quiver(p[:,0], p[:,1], p[:,2], p[:,0] + o[:,0], p[:,1] + o[:,1], p[:,2] + o[:,2], length = 100, normalize = True)
plt.show()
"""
scene = canvas(background=color.white, width=1920, height=1080)
scene.up = vector(0, 0, 1)
scene.forward = vector(1,0,0)
xaxis = cylinder(pos=vec(0,0,0), axis=vec(100,0,0), radius=0.5, color=color.red)
yaxis = cylinder(pos=vec(0,0,0), axis=vec(0,100,0), radius=0.5, color=color.green)
zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,100), radius=0.5, color=color.blue)
k = 1.02
h = 5
text(pos=xaxis.pos + k * xaxis.axis, text='x', height=h, align='center', billboard=True, color=color.black)
text(pos=yaxis.pos + k * yaxis.axis, text='y', height=h, align='center', billboard=True, color=color.black)
text(pos=zaxis.pos + k * zaxis.axis, text='z', height=h, align='center', billboard=True, color=color.black)
b = box(pos = vector(0, 0, 0), size=vector(4, 4, 4), color=color.blue, make_trail=True)
for pos, rot in zip(p, o):
    b.pos = vector(*pos)
    b.up = vector(*rot)
    print(rot)
    sleep(0.0001)
