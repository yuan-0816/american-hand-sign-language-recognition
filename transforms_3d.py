import numpy as np
from scipy.spatial.transform import Rotation as R

class GaussianNoise:
    def __init__(self, amount=0.001):
        self.amount = amount

    def __call__(self, x):
        noise = self.amount * np.random.normal(size=x.shape)
        return x + noise

class Rotate:
    def __init__(self, axis='y', angle = 0.0, prob=1.0):
        if isinstance(angle, list) or isinstance(angle, tuple):
            angle = np.random.uniform(angle[0], angle[1])
        self.angle_in_degree = angle
        self.axis = axis
        self.rotation_matrix = R.from_euler(self.axis, angle, degrees=True).as_matrix()
        self.prob = prob

    def __call__(self, x):
        if np.random.rand() < self.prob:
            rot_points = []
            for i in range(2):
                x[:,  i] = x[:,  i] - 0.5
            for point in x:
                coord = np.dot(self.rotation_matrix, point)
                rot_points.append(coord)
            rot_points = np.array(rot_points)
            for i in range(2):
                rot_points[:, i] = rot_points[:, i] + 0.5

            return rot_points

        return x


class Scale:
    def __init__(self, prob=1.0, axis='x', factor=1.0):
        self.prob = prob
        self.axis = axis
        self.axes = {'x':0, 'y':1, 'z':2}
        self.scale_matrix = np.array([[1, 0.0, 0.0],
                                      [0.0, 1, 0.0],
                                      [0.0, 0.0, 1]])
        self.scale_matrix[self.axes[axis], self.axes[axis]] = factor
        self.delta = (1 - factor) / 2

    def __call__(self, x):
        if np.random.rand() <= self.prob:
            scale_points = []
            for point in x:
                point[self.axes[self.axis]] = point[self.axes[self.axis]] - 0.5
                scaled = np.dot(self.scale_matrix, point)
                scaled[self.axes[self.axis]] = scaled[self.axes[self.axis]] + 0.5
                scale_points.append(scaled)
            return np.array(scale_points)

        return x






