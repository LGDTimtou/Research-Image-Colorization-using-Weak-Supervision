from enum import Enum
import numpy as np
from numpy.random import Generator, PCG64
from scipy.stats import qmc

def _init_gaussian_mask(self, amount, frame_size, sigma=0.2):
    center_y, center_x = int(frame_size / 2), int(frame_size / 2)
    xs = np.clip(self.rng.normal(center_x, sigma * frame_size, amount).astype(int), 0, frame_size - 1)
    ys = np.clip(self.rng.normal(center_y, sigma * frame_size, amount).astype(int), 0, frame_size - 1)
    return np.column_stack((xs, ys))


def _init_random_mask(self, amount, frame_size):
    xs = self.rng.integers(0, frame_size, amount)
    ys = self.rng.integers(0, frame_size, amount)
    return np.column_stack((xs, ys))
    


def _init_poisson_disk_mask(self, amount, frame_size, factor=1.375):
    radius = factor * np.sqrt((frame_size * frame_size) / (amount * np.pi))
    radius_norm = radius / frame_size
    engine = qmc.PoissonDisk(d=2, radius=radius_norm, seed=self.rng)
    samples = engine.integers(l_bounds=[0, 0], u_bounds=[frame_size, frame_size], n=amount)
    return np.column_stack((samples[:, 1], samples[:, 0]))

def _init_grid_mask(self, amount, frame_size):
    amount_sqrt = int(np.sqrt(amount))

    x = np.linspace(0, frame_size - 1, amount_sqrt).astype(int)
    y = np.linspace(0, frame_size - 1, amount_sqrt).astype(int)

    xs, ys = np.meshgrid(x, y)
    return np.column_stack((xs.flatten(), ys.flatten()))

def get_sampling_option(name: str):
    try:
        return SamplingOption[name.upper()]
    except KeyError:
        raise ValueError(f"{name} is not a valid sampling option")


class SamplingOption(Enum):

    GAUSSIAN = ("Gaussian", _init_gaussian_mask, {"sigma": 0.2})
    POISSON = ("Poisson", _init_poisson_disk_mask, {"lambda": 1})
    RANDOM = ("Random", _init_random_mask, {})
    GRID = ("Grid", _init_grid_mask, {})

    def __init__(self, display_name, function, parameters):
        self.display_name = display_name
        self.function = function
        self.parameters = parameters
        self.rng = Generator(PCG64())

    def call_function(self, sampling_amount, frame_size):
        return self.function(self, sampling_amount, frame_size)
    
    
