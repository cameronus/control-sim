import numpy as np
from simulator import Simulator
from vehicle import Vehicle
from visualization import Visualizer

SIM_TIME = 10

POSITION0 = np.array([0, 0, 0])
VELOCITY0 = np.array([0, 0, 0])
ORIENTATION0 = np.array([1, -0.2, 0.3, 0.1])
OMEGA0 = np.array([0, 0, 0])

def main():
    vehicle = Vehicle()
    simulator = Simulator(controller=vehicle, initial_state=np.array([*POSITION0, *VELOCITY0, *ORIENTATION0, *OMEGA0]), sim_time=SIM_TIME)
    simulator.simulate()
    visualization = Visualizer(simulator=simulator)
    print(simulator.state)

if __name__ == "__main__":
    main()
