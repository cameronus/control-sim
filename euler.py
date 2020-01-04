import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *

mass = 1.
inertia_mat = np.matrix([[mass / 6, 0, 0], [0, mass / 6, 0], [0, 0, mass / 6]])
thrust_origin = np.array([0, 0, -0.5])

position = np.zeros((3))
velocity = np.zeros((3))
# velocity = np.array([10., 5., 30.])

orientation = np.quaternion(1, 0, 0, 0)
omega = np.zeros((3))
# omega = np.array([1., 0., -0.2])

t = 0
t_end = 10
dt = 1/100

states = np.zeros((round(t_end / dt) + 1, 16)) # [x, y, z, vx, vy, vz, ow, ox, oy, oz, ωx, ωy, ωz, tx, ty, tz]

def control_alg(): # reducing accuracy of provided data (adding noise to simulate IMU noise)
    return np.array([np.sqrt(1 * 0.5**(t/10)), 0., 10.])

def apply_forces(): # additional forces: drag (shear stress, friction torque), wind
    thrust = control_alg()
    rot_thrust_force = quaternion.rotate_vectors(orientation, thrust)
    force = np.array([0, 0, -9.81]) + rot_thrust_force
    torque = np.cross(thrust_origin, thrust)
    return force, torque, rot_thrust_force

while t <= t_end: # implement ground collision for take-off/landing simulations
    force, torque, rot_thrust = apply_forces()
    states[round(t / dt)] = np.array([*position, *velocity, *quaternion.as_float_array(orientation), *omega, *rot_thrust])
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

scene = canvas(background=color.white, width=1760, height=900) # title='Some title', caption='A caption'
scene.up = vector(0, 0, 1)
scene.forward = vector(-1, -1, -1)
xaxis = cylinder(pos=vec(0, 0, 0), axis=vec(20, 0, 0), radius=0.2, color=color.red)
yaxis = cylinder(pos=vec(0, 0, 0), axis=vec(0, 20, 0), radius=0.2, color=color.green)
zaxis = cylinder(pos=vec(0, 0, 0), axis=vec(0, 0, 20), radius=0.2, color=color.blue)
text(pos=xaxis.pos + 1.02 * xaxis.axis, text='x', height=1, align='center', billboard=True, color=color.black)
text(pos=yaxis.pos + 1.02 * yaxis.axis, text='y', height=1, align='center', billboard=True, color=color.black)
text(pos=zaxis.pos + 1.02 * zaxis.axis, text='z', height=1, align='center', billboard=True, color=color.black)
b = box(pos = vector(0, 0, 0), size=vector(1, 1, 1), color=color.blue, make_trail=True) # trail_type='points', interval=10
scene.camera.follow(b)
v = arrow(pos=vector(0, 0, 0), color=color.yellow)
for state in states:
    position, velocity, orientation, omega, thrust = np.split(state, [3, 6, 10, 13]) # position, velocity, orientation, angular velocity, thrust
    orientation = quaternion.from_float_array(orientation)
    up = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
    b.pos = vector(*position)
    b.up = vector(*up)
    v.pos = vector(*(position + quaternion.rotate_vectors(orientation, thrust_origin)))
    # v.up = vector(*up)
    v.axis = vector(*-thrust / np.linalg.norm(thrust) * 2)
    sleep(0.0001)
