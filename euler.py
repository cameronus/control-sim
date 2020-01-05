import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *
from simple_pid import PID

mass = 1.
edf_force = 14.4207
inertia_mat = np.matrix([[mass / 6, 0, 0], [0, mass / 6, 0], [0, 0, mass / 6]])
thrust_origin = np.array([0, 0, -0.5])

position = np.zeros((3))
velocity = np.zeros((3))
# velocity = np.array([10., 5., 30.])

# orientation = np.quaternion(1, 0, 0, 0)
orientation = np.quaternion(1, -0.9, 0.4, 0)
omega = np.zeros((3))
# omega = np.array([1., 0., -0.2])

t, t_end = 0, 10
dt = 1 / 4000
refresh = 500 # Hz

states = np.zeros((round(t_end / dt) + 1, 19)) # [x, y, z, vx, vy, vz, ow, ox, oy, oz, ωx, ωy, ωz, tx, ty, tz]

tuning = 3, 0.5, 0.6
pid_x = PID(*tuning, setpoint=0)
pid_y = PID(*tuning, setpoint=0)
thrust = np.array([0, 0, edf_force])

def control_alg(acceleration): # reducing accuracy of provided data (adding noise to simulate IMU noise)
    # available: orientation, angular velocity, acceleration
    # mu, sigma = 0, 0.1
    # noise = np.random.normal(mu, sigma, (3))
    # print(round(t, 4))
    # print(quaternion.as_float_array(orientation)[1])
    # print(quaternion.as_euler_angles(orientation))
    # print(omega)
    # print(acceleration)
    # print()

    # rot = quaternion.as_rotation_vector(orientation)
    # rot_mag = np.sqrt(rot[0]**2 + rot[1]**2 + rot[2]**2)
    # control_x = pid_x(rot_mag)
    # control_y = pid_y(rot_mag)
    quat = quaternion.as_float_array(orientation)
    control_x = pid_x(quat[1])
    control_y = pid_y(quat[2])
    # IDEA: use a quaternion to control orientation of thrust vector (which is of magnitude edf_force)
    return np.array([control_x, control_y, np.sqrt(edf_force**2 - control_x**2 - control_y**2)])

def apply_forces(): # external forces: gravity, drag (shear stress, friction torque), wind; TODO: implement torque component of drag
    wind = np.array([0, 0, 0])
    # wind = np.array([1, 2.5, 0])
    drag = 0.5 * 1.225 * 1 * 1 * (velocity - wind) * np.absolute(velocity - wind) # Fd = 1/2 * ρ * Cd * A * v^2
    force = np.array([0, 0, -9.81]) - drag
    return force

while round(t, 6) <= t_end: # TODO: fix buggy time rounding, TODO: implement ground collision for take-off/landing simulations
    ext_force = apply_forces()
    if round(t / dt) % (1 / (dt * refresh)) == 0:
        thrust = control_alg((ext_force + thrust) / mass)
    rot_thrust = quaternion.rotate_vectors(orientation, thrust)

    force = ext_force + rot_thrust
    torque = np.cross(thrust_origin, thrust)

    states[round(t / dt)] = np.array([*position, *velocity, *quaternion.as_float_array(orientation), *omega, *thrust, *rot_thrust])

    velocity += force / mass * dt
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

scale_factor = 40
states = states[::scale_factor]

sleep(1)

for i, state in enumerate(states):
    position, velocity, orientation, omega, thrust, rot_thrust = np.split(state, [3, 6, 10, 13, 16]) # position, velocity, orientation, angular velocity, thrust
    orientation = quaternion.from_float_array(orientation)

    scene.title = f't={round(i * scale_factor * dt, 2)}s<br>position: {np.array_str(position, precision=3)}<br>velocity: {np.array_str(velocity, precision=3)}<br>orientation: {orientation}<br>thrust: {np.array_str(thrust, precision=3)}'

    up = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
    b.pos = vector(*position)
    b.up = vector(*up)
    v.pos = vector(*(position + quaternion.rotate_vectors(orientation, thrust_origin)))
    # v.up = vector(*up) # TODO: set vector to one perpendicular to the axis in the direction closest to the box
    v.axis = vector(*-rot_thrust / np.linalg.norm(rot_thrust) * 1.5)
    sleep(0.000001)
