import numpy as np

class NoiseGenerator:
    def __init__(self, noise_type='gaussian', mu=0.0, sigma=1.0, s0=0, dt=1, seed=42):
        self.noise_type = noise_type
        self.mu = mu
        self.sigma = sigma
        self.s0 = s0
        self.dt = dt
        np.random.seed(seed)  # For reproducibility
        self.n = None

    def generate(self, n):
        self.n=n

        if self.noise_type == 'gaussian':
            return self._gaussian()
        elif self.noise_type == 'brownian':
            return self._brownian()
        elif self.noise_type == 'exponential_brownian':
            return self._exponential_brownian()
        else:
            raise ValueError("Invalid noise type. Choose from 'gaussian', 'brownian', or 'exponential_brownian'.")

    def _gaussian(self):
        """Generates Gaussian (normal) noise."""
        return np.random.normal(self.mu, self.sigma, self.n)

    def _brownian(self):
        """Generates Brownian motion (Wiener process)."""
        w = np.random.normal(self.mu, np.sqrt(self.dt), self.n).cumsum()
        return self.s0 + self.mu * np.arange(self.n) * self.dt + self.sigma * w

    def _exponential_brownian(self):
        """Generates Exponential Brownian Motion (Geometric Brownian Motion)."""
        w = np.random.normal(0, np.sqrt(self.dt), self.n).cumsum()
        time = np.linspace(0, self.n * self.dt, self.n)
        return self.s0 * np.exp((self.mu - 0.5 * self.sigma**2) * time + self.sigma * w)