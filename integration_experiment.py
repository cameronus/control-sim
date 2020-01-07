import numpy as np
import quaternion
from scipy import integrate
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simple_pid import PID
from vpython import *

mass = 1
edf_force = 14.4207
inertia_mat = np.matrix([[mass / 6, 0, 0], [0, mass / 6, 0], [0, 0, mass / 6]])
thrust_origin = np.array([0, 0, -0.5])

tuning = 2, .4, .4
pid_x = PID(*tuning, setpoint=0)
pid_y = PID(*tuning, setpoint=0)
thrust = np.array([0, 0, edf_force])

def control_alg(quat): # reducing accuracy of provided data (adding noise to simulate IMU noise)
    # available: orientation, angular velocity, acceleration
    # mu, sigma = 0, 0.1
    # noise = np.random.normal(mu, sigma, (3))
    # quat = quaternion.as_float_array(orientation)
    control_x = pid_x(quat[1])
    control_y = pid_y(quat[2])
    # IDEA: use a quaternion to control orientation of thrust vector (which is of magnitude edf_forc
    return np.array([control_x, control_y, np.sqrt(edf_force**2 - control_x**2 - control_y**2)])

def apply_forces(velocity): # external forces: gravity, drag (shear stress, friction torque), wind; TODO: implement torque component of drag
    wind = np.array([0, 0, 0])
    # wind = np.array([1, 2.5, 0])
    drag = 0.5 * 1.225 * 1 * 1 * (velocity - wind) * np.absolute(velocity - wind) # Fd = 1/2 * ρ * Cd * A * v^2
    force = np.array([0, 0, -9.81]) - drag
    return force

def sim(t, y):
    velocity = y[3:6]
    orientation = quaternion.from_float_array(y[6:10])
    omega = y[10:13]

    ext_force = apply_forces(velocity)
    # if round(t / dt) % (1 / (dt * refresh)) == 0:
        # thrust = control_alg((ext_force + thrust) / mass)
    rot_thrust = quaternion.rotate_vectors(orientation, thrust)

    force = ext_force + rot_thrust
    torque = np.cross(thrust_origin, thrust)

    ydot = np.empty((13)) # [x, y, z, vx, vy, vz, qw, qx, qy, qz, ox, oy, oz]
    ydot[0:3] = velocity
    ydot[3:6] = force / mass

    ydot[6:10] = quaternion.as_float_array(0.5 * np.quaternion(0, *omega) * orientation)
    rot_matrix = quaternion.as_rotation_matrix(orientation)
    rot_inertia_mat = np.asarray(rot_matrix * inertia_mat * np.transpose(rot_matrix))
    ydot[10:13] = np.matmul(np.linalg.inv(rot_inertia_mat), (torque - np.cross(omega, np.matmul(rot_inertia_mat, omega))))

    return ydot

position0 = np.array([0, 0, 0])
velocity0 = np.array([7, -15, 0])
orientation0 = np.array([1, 0.1, -0.1, 0])
omega0 = np.array([0, 0, 0])

t = 0
control_refresh = 1/200
t_sim = 10

state = np.array([*position0, *velocity0, *orientation0, *omega0])
states = np.zeros((round(t_sim / control_refresh) + 1, 16))

def update():
    global state, t, thrust, solution
    thrust = control_alg(state[6:10])
    solution = integrate.solve_ivp(sim, (0, control_refresh), state) # TODO: check diff integration methods
    state = solution.y.T[-1]
    states[round(t / control_refresh)] = np.array([*state, *thrust])
    t += control_refresh

while round(t, 6) <= t_sim:
    # print('control!')
    update()

# print(states)
print('sol: ', solution)
print('state: ', states[-1])

# print(solution[-1])
# print(solution.y.T[-1])

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(solution.y[0], solution.y[1], solution.y[2]) # t_eval
# plt.show()

scene = canvas(background=color.white, width=1770, height=900)
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

sleep(0.4)
scale_factor = 4
states = states[::scale_factor]
for i, state in enumerate(states):
    position, velocity, orientation_float, omega, thrust = np.split(state, [3, 6, 10, 13]) # position, velocity, orientation, angular velocity
    orientation = quaternion.from_float_array(orientation_float)
    rot_thrust = quaternion.rotate_vectors(orientation, thrust)
    scene.title = (
        f't={round(i * scale_factor * control_refresh, 2)}s<br>'
        f'position: {np.array_str(position, precision=3)}<br>'
        f'velocity: {np.array_str(velocity, precision=3)}<br>'
        f'orientation: {orientation}<br>'
        f'euler: {np.array_str(R.from_quat(orientation_float).as_euler("zyx", degrees=True), precision=0)}<br>'
        f'thrust: {np.array_str(thrust, precision=3)}'
    )

    up = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
    b.pos = vector(*position)
    b.up = vector(*up)
    v.pos = vector(*(position + quaternion.rotate_vectors(orientation, thrust_origin)))
    # v.up = vector(*up) # TODO: set vector to one perpendicular to the axis in the direction closest to the box
    v.axis = vector(*-rot_thrust / np.linalg.norm(rot_thrust) * 1.5)
    # sleep(0.000001)
    sleep(10)

"""
Sample simulation w/out control: (parameters consistent)

Separated integration with control refresh of 200 Hz
[-6.06134090e+00  3.93523367e+01 -1.85370168e+01  4.33912445e+00
 -8.39124449e-01 -5.39413087e+00 -5.23133894e-01 -6.02632114e-01
 -6.02632114e-01 -1.01337204e-18 -6.00000000e-01 -6.00000000e-01
 -8.63963297e-18]
Splitting integration with a 200 Hz control refresh results in a 0.0164% error from "ground truth" (defined as integrating over the whole time period at once)

Separated integration with control refresh of 500 Hz
[-6.06134090e+00  3.93523367e+01 -1.85370168e+01  4.33912445e+00
 -8.39124450e-01 -5.39413087e+00 -5.23133894e-01 -6.02632114e-01
 -6.02632114e-01  8.16799804e-19 -6.00000000e-01 -6.00000000e-01
 -1.08475081e-18]
Same result as with 200 Hz control refresh

One integration
[-6.06351026e+00  3.93458661e+01 -1.85469274e+01  4.33869401e+00
 -8.38694008e-01 -5.39122351e+00 -5.23133889e-01 -6.02632096e-01
 -6.02632096e-01 -8.54667860e-17 -6.00000000e-01 -6.00000000e-01
 -5.59836606e-17]

All integrations using dynamic timestep RK4
"""
