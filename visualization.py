import numpy as np
import quaternion
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from vpython import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class Visualizer:
    def __init__(self, simulator):
        self.FPS = 60
        self.simulator = simulator

        # Interpolate data to create frames
        self.frames = interpolate.interp1d(np.arange(0, self.simulator.NUM_STEPS), self.simulator.state, axis=0)(np.linspace(0, self.simulator.NUM_STEPS - 1, self.FPS * self.simulator.SIM_TIME + 1))

        self.scene = canvas(background=color.white, width=1770, height=900)
        self.scene.up = vector(0, 0, 1)
        self.scene.forward = vector(-1, -1, -1)

        xaxis = cylinder(pos=vector(0, 0, 0), axis=vector(20, 0, 0), radius=0.2, color=color.red)
        yaxis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 20, 0), radius=0.2, color=color.green)
        zaxis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, 20), radius=0.2, color=color.blue)
        text(pos=xaxis.pos + 1.02 * xaxis.axis, text='x', height=1, align='center', billboard=True, color=color.black)
        text(pos=yaxis.pos + 1.02 * yaxis.axis, text='y', height=1, align='center', billboard=True, color=color.black)
        text(pos=zaxis.pos + 1.02 * zaxis.axis, text='z', height=1, align='center', billboard=True, color=color.black)

        self.b = box(pos=vector(0, 0, 0), size=vector(1, 1, 1), color=color.blue, make_trail=True)
        self.v1 = arrow(pos=vector(0, 0, 0), color=color.yellow)
        self.v2 = arrow(pos=vector(0, 0, 0), color=color.yellow)
        self.v3 = arrow(pos=vector(0, 0, 0), color=color.yellow)
        self.v4 = arrow(pos=vector(0, 0, 0), color=color.yellow)
        self.scene.camera.follow(self.b)

        for n in range(self.frames.shape[0]):
            start = time.time_ns()
            position, velocity, orientation_float, omega, control_output = np.split(self.frames[n], [3, 6, 10, 13])
            orientation = quaternion.from_float_array(orientation_float)
            euler = R.from_quat(orientation_float).as_euler('zyx', degrees=True)
            euler[0] *= -1
            euler[2] -= 180
            euler[2] *= -1
            self.scene.title = (
                f't={round(n / self.FPS, 2)}s<br>'
                f'position: {np.array_str(position, precision=3)}<br>'
                f'velocity: {np.array_str(velocity, precision=3)}<br>'
                f'orientation: {orientation}<br>'
                f'euler: {np.array_str(euler, precision=3)}<br>'
                f'control_output: {np.array_str(control_output, precision=3)}'
            )

            up = quaternion.rotate_vectors(orientation, np.array([0, 0, 1]))
            forward = quaternion.rotate_vectors(orientation, np.array([1, 0, 0]))
            self.b.pos = vector(*position)
            self.b.up = vector(*up)
            self.b.axis = vector(*forward)
            for i, vec in enumerate([self.v1, self.v2, self.v3, self.v4]):
                vec.pos = vector(*(position + quaternion.rotate_vectors(orientation, self.simulator.controller.THRUST_ORIGINS[i])))
                vec.up = vector(*forward)
                # length = np.linalg.norm(rot_thrust)
                # thrust_axis = -rot_thrust / length if length > 0 else (0, 0, 0)
                deg = control_output[i]
                vec.axis = vector(*-quaternion.rotate_vectors(orientation, R.from_euler('y' if (i + 1) % 2 == 0 else 'x', deg if i < 2 else -deg, degrees=True).apply(np.array([0, 0, 1]))))
            time.sleep((1 / self.FPS - (time.time_ns() - start) / 1e9) * 0.90568) # Correction factor for visualization speed
            # time.sleep(0.05)
