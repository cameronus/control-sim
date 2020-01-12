import numpy as np
import quaternion
from simple_pid import PID
from scipy.spatial.transform import Rotation as R

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
        self.PID_TUNING = (2.2, 0.6, 0.3) # P, I, D
        # self.VECTORING_BOUNDS = (-15, 15) # degrees
        self.VECTORING_BOUNDS = (-15, 15) # degrees
        self.VECTORING_SPEED = 30 # degrees per second

        self.pid_x = PID(*self.PID_TUNING, setpoint=0)
        self.pid_y = PID(*self.PID_TUNING, setpoint=0)
        self.pid_z = PID(*self.PID_TUNING, setpoint=0)
        self.pid_x.output_limits = self.VECTORING_BOUNDS
        self.pid_y.output_limits = self.VECTORING_BOUNDS
        self.pid_z.output_limits = self.VECTORING_BOUNDS

    def update_setpoints(self, t):
        return np.array([0, 0, 44, self.EDF_THRUST * (t / 15)**(1/6)])

    def update_inputs(self, t, acceleration, orientation, omega):
        euler = R.from_quat(orientation).as_euler('zyx', degrees=True)
        euler[0] *= -1
        euler[2] -= 180
        euler[2] *= -1
        orientation = quaternion.from_float_array(orientation)

        setpoints = self.update_setpoints(t)
        self.pid_x.setpoint, self.pid_y.setpoint, self.pid_z.setpoint, thrust = setpoints
        angle_x = self.pid_x(euler[1])
        angle_y = self.pid_y(euler[0])
        abs_err = abs(self.pid_z.setpoint - euler[2])
        if abs_err < 180:
            err = abs_err
        else:
            err = abs_err - 360
        angle_z = self.pid_z(err)

        # TODO: add height measurement, add noise to measurements, ground collision physics, throttle pid

        # print(euler)
        # print(angle_x, angle_y, angle_z)

        vane_angles = np.array([-angle_y - angle_z, -angle_x - angle_z, angle_y - angle_z, angle_x - angle_z])

        vane_vec = np.array([0, 0, thrust / 4])
        vane_force = np.zeros((4, 3))
        for i, deg in enumerate(vane_angles):
            vane_force[i] = R.from_euler('y' if (i + 1) % 2 == 0 else 'x', deg if i < 2 else -deg, degrees=True).apply(vane_vec)
        force = quaternion.rotate_vectors(orientation, np.sum(vane_force, axis=0))
        torque = np.sum(np.cross(self.THRUST_ORIGINS, vane_force), axis=0)
        torque.real[abs(torque.real) < np.finfo(np.float).eps] = 0.0

        return np.array([*vane_angles, thrust]), force, torque
