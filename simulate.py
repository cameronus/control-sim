import numpy as np
import quaternion
from scipy import integrate
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from vpython import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simple_pid import PID

# Vehicle constants
MASS = 1 # kg
DRAG_COEFF = 1 # TODO: revamp drag calculations
AREA = 1 # m^2
EDF_THRUST = 14.4207 # N
THRUST_ORIGIN = np.array([0, 0, -0.5]) # Origin of thrust vectoring
INERTIA_TENSOR = np.matrix([[MASS / 6, 0, 0],
                            [0, MASS / 6, 0],
                            [0, 0, MASS / 6]])

# Environment constants
GRAVITY = 9.81 # m/s^2
WIND = np.array([0, 0, 0]) # m/s
FLUID_DENSITY = 1.225 # kg/m^3

# Control constants
CONTROL_FREQ = 200 # Hz
PID_TUNING = (2, 0.4, 0.4) # P, I, D
VECTORING_BOUNDS = (-15, 15) # degrees

# Simulation constants
SIM_TIME = 10 # seconds
NUM_STEPS = round(SIM_TIME * CONTROL_FREQ) + 1
POSITION0 = np.array([0, 0, 0])
VELOCITY0 = np.array([0, 0, 0])
ORIENTATION0 = np.array([1, 0, 0, 0])
OMEGA0 = np.array([0, 0, 0])

# Visualization constants
FPS = 60

# State variables
thrust = np.array([0, 0, EDF_THRUST])
state = np.zeros((NUM_STEPS, 16))

# Initialize state
# state[0] = np.array([*POSITION0, *VELOCITY0, *ORIENTATION0, *OMEGA0, *thrust])
state[0][0:13] = np.array([*POSITION0, *VELOCITY0, *ORIENTATION0, *OMEGA0])

print(f'Simulating {SIM_TIME} seconds with a control frequency of {CONTROL_FREQ} Hz and a visualization FPS of {FPS}.')

def state_dot(t, state):
    velocity = state[3:6]
    orientation = quaternion.from_float_array(state[6:10])
    omega = state[10:13]

    drag = 0.5 * FLUID_DENSITY * DRAG_COEFF * AREA * (velocity - WIND) * np.absolute(velocity - WIND)

    force = np.array([0, 0, -GRAVITY]) - drag + quaternion.rotate_vectors(orientation, thrust)
    torque = np.cross(THRUST_ORIGIN, thrust)

    state_dot = np.empty((13)) # [x, y, z, vx, vy, vz, qw, qx, qy, qz, ox, oy, oz]
    state_dot[0:3] = velocity
    state_dot[3:6] = force / MASS

    state_dot[6:10] = quaternion.as_float_array(0.5 * np.quaternion(0, *omega) * orientation)
    rotation_matrix = quaternion.as_rotation_matrix(orientation)
    rot_inertia_tensor = np.asarray(rotation_matrix * INERTIA_TENSOR * np.transpose(rotation_matrix))
    state_dot[10:13] = np.matmul(np.linalg.inv(rot_inertia_tensor), (torque - np.cross(omega, np.matmul(rot_inertia_tensor, omega))))

    return state_dot

def control_alg(acceleration, orientation, omega):
    # print('Control Algo')
    # print('acceleration =', acceleration)
    # print('orientation =', orientation)
    # print('omega =', omega)
    return np.array([0, 0, EDF_THRUST])

for step in range(1, NUM_STEPS):
    acceleration = state_dot(step / CONTROL_FREQ, state[step - 1])[3:6]
    thrust = control_alg(acceleration, state[step - 1][6:10], state[step - 1][10:13])
    solution = integrate.solve_ivp(state_dot, (0, 1 / CONTROL_FREQ), state[step - 1][:13])
    state[step] = np.concatenate([solution.y.T[-1], thrust])

# Visualization frames
frames = interpolate.interp1d(np.arange(0, NUM_STEPS), state, axis=0)(np.linspace(0, NUM_STEPS - 1, FPS * SIM_TIME + 1))
# TODO: Refactor array operations in state_dot function

scene = canvas(background=color.white, width=1770, height=900)
scene.up = vector(0, 0, 1)
scene.forward = vector(-1, -1, -1)

xaxis = cylinder(pos=vec(0, 0, 0), axis=vec(20, 0, 0), radius=0.2, color=color.red)
yaxis = cylinder(pos=vec(0, 0, 0), axis=vec(0, 20, 0), radius=0.2, color=color.green)
zaxis = cylinder(pos=vec(0, 0, 0), axis=vec(0, 0, 20), radius=0.2, color=color.blue)
text(pos=xaxis.pos + 1.02 * xaxis.axis, text='x', height=1, align='center', billboard=True, color=color.black)
text(pos=yaxis.pos + 1.02 * yaxis.axis, text='y', height=1, align='center', billboard=True, color=color.black)
text(pos=zaxis.pos + 1.02 * zaxis.axis, text='z', height=1, align='center', billboard=True, color=color.black)

b = box(pos=vector(0, 0, 0), size=vector(1, 1, 1), color=color.blue, make_trail=True)
v = arrow(pos=vector(0, 0, 0), color=color.yellow)
scene.camera.follow(b)

for i, frame in enumerate(frames):
    position, velocity, orientation_float, omega, thrust = np.split(frame, [3, 6, 10, 13])
    orientation = quaternion.from_float_array(orientation_float)
    rot_thrust = quaternion.rotate_vectors(orientation, thrust)
    scene.title = (
        f't={round(i / FPS, 2)}s<br>'
        f'position: {np.array_str(position, precision=3)}<br>'
        f'velocity: {np.array_str(velocity, precision=3)}<br>'
        f'orientation: {orientation}<br>'
        f'euler: {np.array_str(R.from_quat(orientation_float).as_euler("zyx", degrees=True), precision=0)}<br>'
        f'thrust: {np.array_str(thrust, precision=3)}'
    )

    up = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
    b.pos = vector(*position)
    b.up = vector(*up)
    v.pos = vector(*(position + quaternion.rotate_vectors(orientation, THRUST_ORIGIN)))
    # v.up = vector(*up) # TODO: set vector to one perpendicular to the axis in the direction closest to the box
    print(rot_thrust)
    print(np.linalg.norm(rot_thrust))
    length = np.linalg.norm(rot_thrust)
    thrust_axis = -rot_thrust / length * 1.5 if length > 0 else [0, 0, 0]
    v.axis = vector(*thrust_axis)
    # sleep(0.016667)
    sleep(10)
