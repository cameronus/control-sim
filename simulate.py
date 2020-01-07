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
state[0][0:13] = np.array([*POSITION0, *VELOCITY0, *ORIENTATION0, *OMEGA0])

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
    # print(step)
    acceleration = state_dot(step / CONTROL_FREQ, state[step - 1])[3:6]
    thrust = control_alg(acceleration, state[step - 1][6:10], state[step - 1][10:13])
    solution = integrate.solve_ivp(state_dot, (0, 1 / CONTROL_FREQ), state[step - 1][:13])
    state[step] = np.concatenate([solution.y.T[-1], thrust])

# Visualization frames
frames = interpolate.interp1d(np.arange(0, NUM_STEPS), state, axis=0)(np.linspace(0, NUM_STEPS - 1, FPS * SIM_TIME))
# TODO: Refactor array operations in state_dot function
