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



t, t_end = 0, 10
dt = 1 / 1000
refresh = 500 # Hz

states = np.zeros((round(t_end / dt) + 1, 16)) # [x, y, z, vx, vy, vz, ow, ox, oy, oz, ωx, ωy, ωz, tx, ty, tz]


thrust = np.array([0, 0, 0])
acceleration = np.array([0, 0, 0])

def control_alg(): # reducing accuracy of provided data (adding noise to simulate IMU noise)
    # available: orientation, angular velocity, acceleration
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, (3))
    print(round(t, 4))
    print(orientation)
    print(omega)
    print(acceleration)
    print()
    return np.array([0., -0.01, 12.])

def apply_forces(): # additional forces: drag (shear stress, friction torque), wind
    global thrust
    if round(t / dt) % (1 / (dt * refresh)) == 0:
        thrust = control_alg()
    rot_thrust = quaternion.rotate_vectors(orientation, thrust)

    wind = np.array([-1, -2.5, 0])
    drag = 0.5 * 1.225 * 1 * 1 * (velocity - wind) * np.absolute(velocity - wind) # Fd = 1/2 * ρ * Cd * A * v^2

    force = np.array([0, 0, -9.81]) - drag + rot_thrust
    torque = np.cross(thrust_origin, thrust)
    return force, torque, rot_thrust

while t <= t_end: # implement ground collision for take-off/landing simulations
    force, torque, rot_thrust = apply_forces()

    states[round(t / dt)] = np.array([*position, *velocity, *quaternion.as_float_array(orientation), *omega, *rot_thrust])

    acceleration = force / mass
    velocity += acceleration * dt
    position += velocity * dt

    orientation += orientation * np.quaternion(0, *omega) * 0.5 * dt
    rot_matrix = quaternion.as_rotation_matrix(orientation)
    rot_inertia_mat = np.asarray(rot_matrix * inertia_mat * np.transpose(rot_matrix))
    omega += np.matmul(np.linalg.inv(rot_inertia_mat), (torque - np.cross(omega, np.matmul(rot_inertia_mat, omega)))) * dt

    t += dt

scene = canvas(background=color.white, width=1770, height=900) # title='Some title', caption='A caption'
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

scale_factor = 10
states = states[::scale_factor]

for i, state in enumerate(states):
    position, velocity, orientation, omega, thrust = np.split(state, [3, 6, 10, 13]) # position, velocity, orientation, angular velocity, thrust

    scene.title = f't={round(i * scale_factor * dt, 2)}s<br>position: {np.array_str(position, precision=3)}<br>velocity: {np.array_str(velocity, precision=3)}'

    orientation = quaternion.from_float_array(orientation)
    up = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
    b.pos = vector(*position)
    b.up = vector(*up)
    v.pos = vector(*(position + quaternion.rotate_vectors(orientation, thrust_origin)))
    v.axis = vector(*-thrust / np.linalg.norm(thrust) * 2)
    # v.up = vector(*up)
    sleep(0.000001)
