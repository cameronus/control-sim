import numpy as np
import quaternion
from simple_pid import PID
from scipy.spatial.transform import Rotation as R

class Vehicle:
    def __init__(self):
        # Vehicle constants (r=0.09m, l=0.24m)
        self.RADIUS = 0.09
        self.LENGTH = 0.36
        self.MASS = 1 # kg
        self.DRAG_COEFF = 1
        self.AREA = 0.05 # m^2
        self.EDF_THRUST = 14.42 # N
        self.THRUST_ORIGINS = np.array([[0.07, 0, -self.LENGTH / 2],
                                        [0, 0.07, -self.LENGTH / 2],
                                        [-0.07, 0, -self.LENGTH / 2],
                                        [0, -0.07, -self.LENGTH / 2]])
        self.INERTIA_TENSOR = np.matrix([[self.MASS * 0.006825, 0, 0],
                                         [0, self.MASS * 0.006825, 0],
                                         [0, 0, self.MASS * 0.00405]])

        # Control constants
        self.CONTROL_FREQ = 200 # Hz
        self.PID_TILT = (1, 0.1, 0.1) # P, I, D
        self.PID_YAW = (0.3, 0, 0.15) # P, I, D
        self.PID_ALT = (2.4, 1.02, 1.4) # P, I, D (tuned)
        self.VECTORING_BOUNDS = tuple([x / 2 for x in (-15, 15)]) # degrees
        self.VECTORING_SPEED = 600 # degrees per second
        self.THRUST_LOST_TO_VECTORING = 0.7 # percent of thrust diverted by vectoring in -z direction

        self.pid_x = PID(*self.PID_TILT, setpoint=0)
        self.pid_y = PID(*self.PID_TILT, setpoint=0)
        self.pid_z = PID(*self.PID_YAW, setpoint=0)
        self.pid_thrust = PID(*self.PID_ALT, setpoint=0)
        self.pid_x.output_limits = self.VECTORING_BOUNDS
        self.pid_y.output_limits = self.VECTORING_BOUNDS
        self.pid_z.output_limits = self.VECTORING_BOUNDS
        self.pid_thrust.output_limits = (0, self.EDF_THRUST)

    def update_setpoints(self, t):
        return np.array([0, 0, 0, 10])

    def update_inputs(self, t, prev_output, altitude, acceleration, orientation, omega):
        euler = R.from_quat(orientation).as_euler('zyx', degrees=True)
        euler[0] *= -1
        euler[2] -= 180
        euler[2] *= -1
        orientation = quaternion.from_float_array(orientation)

        self.pid_x.setpoint, self.pid_y.setpoint, z_setpoint, self.pid_thrust.setpoint = self.update_setpoints(t)
        angle_x = self.pid_x(euler[1])
        angle_y = self.pid_y(euler[0])
        abs_err = (euler[2] - z_setpoint) % 360
        err = abs_err - (0 if abs_err < 180 else 360)
        angle_z = self.pid_z(err)
        thrust = self.pid_thrust(altitude)

        # print(euler[2])
        print()

        # TODO: FIX YAW, revamp visualization/graphing, deadband, add noise to measurements, ground collision physics, manually implementing PID, add torque from edf

        # vane_angles = np.array([-angle_y - angle_z, -angle_x - angle_z, angle_y - angle_z, angle_x - angle_z])
        desired_vane_angles = np.array([-angle_y - angle_z, -angle_x - angle_z, angle_y - angle_z, angle_x - angle_z])
        vane_angles = np.clip(desired_vane_angles - prev_output[:4], -self.VECTORING_SPEED / self.CONTROL_FREQ, self.VECTORING_SPEED / self.CONTROL_FREQ) + prev_output[:4]

        corrected_thrust = min(thrust / np.cos(np.average(vane_angles) * np.pi / 180), self.EDF_THRUST)
        vane_vec = np.array([0, 0, corrected_thrust * self.THRUST_LOST_TO_VECTORING / 4])
        vane_force = np.zeros((4, 3))
        for i, deg in enumerate(vane_angles):
            vane_force[i] = R.from_euler('y' if (i + 1) % 2 == 0 else 'x', deg if i < 2 else -deg, degrees=True).apply(vane_vec)
        force = quaternion.rotate_vectors(orientation, np.sum(vane_force, axis=0)) + np.array([0, 0, (1 - self.THRUST_LOST_TO_VECTORING) * thrust])
        torque = np.sum(np.cross(self.THRUST_ORIGINS, vane_force), axis=0)
        torque.real[abs(torque.real) < np.finfo(np.float).eps] = 0.0

        return np.array([*vane_angles, thrust]), force, torque
