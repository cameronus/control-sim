import numpy as np
import quaternion
from simple_pid import PID

class Vehicle:
    def __init__(self):
        # Vehicle constants
        self.MASS = 1 # kg
        self.DRAG_COEFF = 1
        self.AREA = 1 # m^2
        self.EDF_THRUST = 14.42 # N
        self.THRUST_ORIGINS = np.array([[0.25, 0, -0.5],
                                        [0, 0.25, -0.5],
                                        [-0.25, 0, -0.5],
                                        [0, -0.25, -0.5]])
        self.INERTIA_TENSOR = np.matrix([[self.MASS / 6, 0, 0],
                                         [0, self.MASS / 6, 0],
                                         [0, 0, self.MASS / 6]])

        # Control constants
        self.CONTROL_FREQ = 200 # Hz
        self.INITIAL_STATE = np.array([0, 0, 0, 0, 0])
        self.PID_TUNING = (2, 0.4, 0.2) # P, I, D
        self.VECTORING_BOUNDS = (-15, 15) # degrees
        self.VECTORING_SPEED = 30 # degrees per second

        self.pid_x = PID(*self.PID_TUNING, setpoint=0)
        self.pid_y = PID(*self.PID_TUNING, setpoint=0)
        self.pid_x.output_limits = self.VECTORING_BOUNDS
        self.pid_y.output_limits = self.VECTORING_BOUNDS

    def update_setpoints(self):
        return np.array([0, 0, 0])

    def update_inputs(self, acceleration, orientation, omega):
        # euler = R.from_quat(orientation).as_euler('zyx', degrees=True)
        # euler[0] *= -1
        # euler[2] -= 180
        # euler[2] *= -1
        # angle_x = pid_x(euler[1])
        # angle_y = pid_y(euler[0])
        #
        # thrust = np.array([np.sin(-angle_x * np.pi / 180) * EDF_THRUST, np.sin(angle_y * np.pi / 180) * EDF_THRUST, np.sqrt(EDF_THRUST**2 - (np.sin(angle_x * np.pi / 180) * EDF_THRUST)**2 - (np.sin(angle_y * np.pi / 180) * EDF_THRUST)**2)])

        # orientation = quaternion.from_float_array(orientation)

        # control_force = quaternion.rotate_vectors(orientation, thrust)
        # control_torque = np.cross(THRUST_ORIGIN, thrust)

        # return thrust
        # return np.array([15, 0, -15, 0, 0])
        setpoints = self.update_setpoints()
        control_input = np.array([15, 0, -15, 0, self.EDF_THRUST])
        force = np.array([0, 0, 0])
        torque = np.array([0, 0, 0])
        return control_input, force, torque
