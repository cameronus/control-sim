import numpy as np
import quaternion
from scipy import integrate

class Simulator:
    def __init__(self, controller, initial_state, sim_time):
        # Environment constants
        self.GRAVITY = 9.81 # m/s^2
        self.WIND = np.array([0, 0, 0]) # m/s
        self.FLUID_DENSITY = 1.225 # kg/m^3

        # Set controller
        self.controller = controller

        # Calculate number of simulation steps
        self.SIM_TIME = sim_time
        self.NUM_STEPS = round(self.SIM_TIME * self.controller.CONTROL_FREQ) + 1

        # Initialize simulation and control states
        self.state = np.zeros((self.NUM_STEPS, 13 + self.controller.INITIAL_STATE.shape[0]))
        self.state[0] = np.array([*initial_state, *self.controller.INITIAL_STATE])
        self.control_force = np.array([0, 0, 0])
        self.control_torque = np.array([0, 0, 0])

        print(f'Simulating {self.SIM_TIME} seconds with a control frequency of {self.controller.CONTROL_FREQ} Hz.')

    def state_dot(self, t, state):
        velocity = state[3:6]
        orientation = quaternion.from_float_array(state[6:10])
        omega = state[10:]

        drag = 0.5 * self.FLUID_DENSITY * self.controller.DRAG_COEFF * self.controller.AREA * (velocity - self.WIND) * np.absolute(velocity - self.WIND)

        force = np.array([0, 0, -self.GRAVITY]) - drag + self.control_force
        torque = self.control_torque

        state_dot = np.empty((13)) # [x, y, z, vx, vy, vz, qw, qx, qy, qz, ox, oy, oz]
        state_dot[0:3] = velocity
        state_dot[3:6] = force / self.controller.MASS

        state_dot[6:10] = quaternion.as_float_array(0.5 * np.quaternion(0, *omega) * orientation)
        rotation_matrix = quaternion.as_rotation_matrix(orientation)
        rot_inertia_tensor = np.asarray(rotation_matrix * self.controller.INERTIA_TENSOR * np.transpose(rotation_matrix))
        state_dot[10:] = np.matmul(np.linalg.inv(rot_inertia_tensor), (torque - np.cross(omega, np.matmul(rot_inertia_tensor, omega))))

        return state_dot

    def simulate(self):
        for step in range(1, self.NUM_STEPS):
            t = step / self.controller.CONTROL_FREQ
            acceleration = self.state_dot(t, self.state[step - 1][:13])[3:6]
            control_input, self.control_force, self.control_torque = self.controller.update_inputs(t, acceleration, self.state[step - 1][6:10], self.state[step - 1][10:13])
            solution = integrate.solve_ivp(self.state_dot, (0, 1 / self.controller.CONTROL_FREQ), self.state[step - 1][:13])
            self.state[step] = np.concatenate([solution.y.T[-1], control_input])
