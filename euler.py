import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *

mass = 1.
inertia_mat = np.matrix([[mass / 6, 0, 0], [0, mass / 6, 0], [0, 0, mass / 6]])

position = np.zeros((3))
velocity = np.zeros((3))
# velocity = np.array([10., 5., 30.])

orientation = np.quaternion(1, 0, 0, 0)
omega = np.zeros((3))
# omega = np.array([1., 0., -0.2])

t = 0
t_end = 10
dt = 1/100

num_steps = round(t_end / dt) + 1
cs = lambda: round(t / dt)

position_hist = np.zeros((num_steps, 3))
orientation_hist = np.empty((num_steps), dtype=np.quaternion)
force_hist = np.zeros((num_steps, 3))

def apply_forces():
    gravity = np.array([0., 0., -9.81])
    # push = np.array([0.2, 0.1, 0.])
    push = np.array([np.cos(20 * t), np.sin(20 * t), 9.81]) # issue: push force is applied linearly no matter the orientation
    force_hist[cs()] = push
    force = gravity + push
    torque = np.cross(push, np.array([0, 0, -0.5]))
    return force, torque

while t <= t_end:
    position_hist[cs()] = position
    orientation_hist[cs()] = orientation
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
xaxis = cylinder(pos=vec(0,0,0), axis=vec(20,0,0), radius=0.2, color=color.red)
yaxis = cylinder(pos=vec(0,0,0), axis=vec(0,20,0), radius=0.2, color=color.green)
zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,20), radius=0.2, color=color.blue)
k = 1.02
h = 1
text(pos=xaxis.pos + k * xaxis.axis, text='x', height=h, align='center', billboard=True, color=color.black)
text(pos=yaxis.pos + k * yaxis.axis, text='y', height=h, align='center', billboard=True, color=color.black)
text(pos=zaxis.pos + k * zaxis.axis, text='z', height=h, align='center', billboard=True, color=color.black)
b = box(pos = vector(0, 0, 0), size=vector(1, 1, 1), color=color.blue, make_trail=True)
scene.camera.follow(b)
v = arrow(pos=vector(0, 0, 0), color=color.yellow)
for pos, rot, force in zip(position_hist, orientation_hist, force_hist):
    # print(rot)
    up = quaternion.rotate_vectors(rot, np.array([0, 0, 1]))

    b.pos = vector(*pos)
    b.up = vector(*up)

    thrust_dir = quaternion.rotate_vectors(rot, np.array([0, 0, -0.5]))
    thrust = quaternion.rotate_vectors(rot, force)

    v.pos = vector(*(pos + thrust_dir))
    # v.up = vector(*up)
    v.axis = vector(*-thrust)
    sleep(0.0001)
