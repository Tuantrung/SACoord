from rl.random import GaussianWhiteNoiseProcess

sigma = 0.3
mu = 0.0

random_process = GaussianWhiteNoiseProcess(sigma=sigma, mu=mu, size=5)

noise = random_process.sample()
print(noise)
print(sum(noise))