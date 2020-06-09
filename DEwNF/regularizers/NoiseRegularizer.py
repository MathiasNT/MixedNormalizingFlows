import pyro.distributions as dist
import torch


class NoiseRegularizer(object):
    def __init__(self, discrete_dims, h, cuda):
        self.h = h
        self.discrete_dims = discrete_dims
        self.cuda = cuda

    def add_noise(self, z):
        noise_dist = self._create_noise_dist(z)
        xi = noise_dist.sample()
        z_hat = z + self.h * xi
        return z_hat

    def _create_noise_dist(self, z):
        means = torch.zeros(z.shape)
        scales = torch.ones(z.shape)

        # Set scale to 0 to avoid adding noise to discrete dims
        if self.discrete_dims is not None:
            for dim in self.discrete_dims:
                scales[:, dim] = 0

        if self.cuda:
            noise_dist = dist.Normal(means.cuda(), scales.cuda())
        else:
            noise_dist = dist.Normal(means, scales)

        return noise_dist


def rule_of_thumb_noise_schedule(n, d, sigma):
    ex = -1 / (4 + d)
    h = 1.06 * sigma * n ** ex
    return h


def approx_rule_of_thumb_noise_schedule(n, d, sigma):
    ex = -1 / (4 + d)
    h = n ** ex
    return h


def square_root_noise_schedule(n, d, sigma):
    ex = -1 / (1 + d)
    h = n ** ex
    return h


def no_regularization_schedule(n, d, sigma):
    return 0


def constant_regularization_schedule(n ,d, sigma):
    # I am aware this is a hacky implementation but it worked the best with my notebooks
    return sigma
