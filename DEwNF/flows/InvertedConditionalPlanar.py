# Based upon implementation from Uber Technologies - Their trademark is below
# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
import torch.nn.functional as F

from torch.distributions import Transform
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.nn import DenseNN
import torch.nn as nn


# Class below has been inverted to work in a density estimation setting.
class InvertedConditionedPlanar(Transform):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, bias=None, u=None, w=None):
        super().__init__(cache_size=1)
        self.bias = bias
        self.u = u
        self.w = w
        self._cached_logDetJ = None

    # This method ensures that torch(u_hat, w) > -1, required for invertibility
    def u_hat(self, u, w):
        alpha = torch.matmul(u.unsqueeze(-2), w.unsqueeze(-1)).squeeze(-1)
        a_prime = -1 + F.softplus(alpha)
        return u + (a_prime - alpha) * w.div(w.pow(2).sum(dim=-1, keepdim=True))

    def _call(self, x):  # Note that this is the original codes inverse, the docstring and error is left as original.
        raise KeyError("ConditionedPlanar object expected to find key in intermediates cache but didn't")

    def _inverse(self, y):  # Note that this is the original codes call, x and y have been interchanged to keep notation

        # x ~ (batch_size, dim_size, 1)
        # w ~ (batch_size, 1, dim_size)
        # bias ~ (batch_size, 1)
        act = torch.tanh(torch.matmul(self.w.unsqueeze(-2), y.unsqueeze(-1)).squeeze(-1) + self.bias)
        u_hat = self.u_hat(self.u, self.w)
        x = y + u_hat * act

        psi_z = (1. - act.pow(2)) * self.w
        self._cached_logDetJ = torch.log(
            torch.abs(1 + torch.matmul(psi_z.unsqueeze(-2), u_hat.unsqueeze(-1)).squeeze(-1).squeeze(-1)))
        return x

    def log_abs_det_jacobian(self, x, y):  # The sign have been switched on the logDetJ to match the inverted flow

        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self._inverse(y)

        return -self._cached_logDetJ


# Class below is as in original
class InvertedConditionalPlanar(ConditionalTransformModule):

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def condition(self, context):
        bias, u, w = self.nn(context)
        return InvertedConditionedPlanar(bias, u, w)


# Function below is as in original
def inverted_conditional_planar(input_dim, context_dim, hidden_dims=None):

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    nn = DenseNN(context_dim, hidden_dims, param_dims=[1, input_dim, input_dim])
    return InvertedConditionalPlanar(nn)


class InvertedPlanar(InvertedConditionedPlanar, TransformModule):

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, input_dim):
        super().__init__()

        self.bias = nn.Parameter(torch.Tensor(1,))
        self.u = nn.Parameter(torch.Tensor(input_dim,))
        self.w = nn.Parameter(torch.Tensor(input_dim,))
        self.input_dim = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u.size(0))
        self.w.data.uniform_(-stdv, stdv)
        self.u.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()