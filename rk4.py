import numpy as np
import quaternion
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simple_pid import PID
from vpython import *

t_sim = 10
t_step = 1000

mass = 1
edf_force = 14.4207
inertia_mat = np.matrix([[mass / 6, 0, 0], [0, mass / 6, 0], [0, 0, mass / 6]])
thrust_origin = np.array([0, 0, -0.5])

tuning = 3, 0.5, 0.6
pid_x = PID(*tuning, setpoint=0)
pid_y = PID(*tuning, setpoint=0)
thrust = np.array([0.02, -0.02, edf_force])

def control_alg(acceleration): # reducing accuracy of provided data (adding noise to simulate IMU noise)
    # available: orientation, angular velocity, acceleration
    # mu, sigma = 0, 0.1
    # noise = np.random.normal(mu, sigma, (3))
    quat = quaternion.as_float_array(orientation)
    control_x = pid_x(quat[1])
    control_y = pid_y(quat[2])
    # IDEA: use a quaternion to control orientation of thrust vector (which is of magnitude edf_forc
    return np.array([control_x, control_y, np.sqrt(edf_force**2 - control_x**2 - control_y**2)])

def apply_forces(velocity): # external forces: gravity, drag (shear stress, friction torque), wind; TODO: implement torque component of drag
    # wind = np.array([0, 0, 0])
    wind = np.array([1, 2.5, 0])
    drag = 0.5 * 1.225 * 1 * 1 * (velocity - wind) * np.absolute(velocity - wind) # Fd = 1/2 * ρ * Cd * A * v^2
    force = np.array([0, 0, -9.81]) - drag
    return force

def sim(t, y):
    print(t)
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
velocity0 = np.array([7, -15, -20])
orientation0 = np.array([1, 0, 0, 0])
omega0 = np.array([0, 0, 0])
y0 = np.array([*position0, *velocity0, *orientation0, *omega0])

solution = solve_ivp(sim, (0, t_sim), y0, t_eval=np.linspace(0, t_sim, t_sim * t_step))
# solution = odeint(sim, y0, np.linspace(0, t_sim, t_sim * t_step), tfirst=True)

# print(solution[-1])
print(solution.y.T[-1])

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

for i, state in enumerate(solution.y.T):
    position, velocity, orientation, omega = np.split(state, [3, 6, 10]) # position, velocity, orientation, angular velocity
    orientation = quaternion.from_float_array(orientation)

    # scene.title = f't={round(i * scale_factor * dt, 2)}s<br>position: {np.array_str(position, precision=3)}<br>velocity: {np.array_str(velocity, precision=3)}<br>orientation: {orientation}<br>thrust: {np.array_str(thrust, precision=3)}'

    up = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
    b.pos = vector(*position)
    b.up = vector(*up)
    # v.pos = vector(*(position + quaternion.rotate_vectors(orientation, thrust_origin)))
    # v.up = vector(*up) # TODO: set vector to one perpendicular to the axis in the direction closest to the box
    # v.axis = vector(*-rot_thrust / np.linalg.norm(rot_thrust) * 1.5)
    sleep(0.000001)
