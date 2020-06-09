# This is the conditioned AffineCoupling flow which is pretty much the pyro AffineCoupling with the context added in.
# I have commented all the places the code is changed to make it conditional.
# The original code is under Apache 2.0 license - The original code trademak and license is below
# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from pyro.distributions import TransformModule, ConditionalTransformModule
from torch.distributions import constraints

from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.nn import DenseNN

from ..nns import DropoutDenseNN


class ConditionedAffineCoupling2(TransformModule):

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, split_dim, hypernet, rich_context, log_scale_min_clip=-5., log_scale_max_clip=3.):
        super().__init__(cache_size=1)
        self.split_dim = split_dim
        self.hypernet = hypernet
        self.rich_context = rich_context # context added when conditioning
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]
        rich_context = self.rich_context.expand(x1.shape[0], -1) # Expand the context to have same dimension as data (only used when same context for all data points)
        x_cov = torch.cat((x1, rich_context), dim = 1) # Concat data and context
        mean, log_scale = self.hypernet(x_cov) # send both data and context into conditioner
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        y1 = x1
        y2 = torch.exp(log_scale) * x2 + mean
        return torch.cat([y1, y2], dim=-1)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """
        y1, y2 = y[..., :self.split_dim], y[..., self.split_dim:]
        x1 = y1

        rich_context = self.rich_context.expand(x1.shape[0], -1) # expand context
        x_cov = torch.cat((x1, rich_context), dim = 1) # concat to data
        mean, log_scale = self.hypernet(x_cov) # send into conditioner

        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        x2 = (y2 - mean) * torch.exp(-log_scale)
        return torch.cat([x1, x2], dim=-1)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if self._cached_log_scale is not None and x is x_old and y is y_old:
            log_scale = self._cached_log_scale
        else:
            x1 = x[..., :self.split_dim]
            rich_context = self.rich_context.expand(x1.shape[0], -1)  # expand context
            x_cov = torch.cat((x1, rich_context), dim=1)  # cat to data
            _, log_scale = self.hypernet(x_cov)  # send to conditioner
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)


class ConditionalAffineCoupling2(ConditionalTransformModule):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, split_dim, hypernet):
        super(ConditionalAffineCoupling2, self).__init__()
        self.hypernet = hypernet
        self.split_dim = split_dim

    def condition(self, rich_context):
        rich_context = rich_context
        return ConditionedAffineCoupling2(self.split_dim, self.hypernet, rich_context)


def conditional_affine_coupling2(input_dim, context_dim, hidden_dims=None, split_dim=None,
                                 rich_context_dim=None, dropout=None, **kwargs):
    if split_dim is None:
        split_dim = input_dim // 2
    if hidden_dims is None:
        hidden_dims = [10 * input_dim]

    if rich_context_dim is None:
        rich_context_dim = 5 * context_dim

    if dropout is None:
        hypernet = DenseNN(split_dim + rich_context_dim, hidden_dims, [input_dim - split_dim, input_dim - split_dim])
    else:
        hypernet = DropoutDenseNN(input_dim=split_dim + rich_context_dim,
                                  hidden_dims=hidden_dims,
                                  dropout=dropout,
                                  param_dims=[input_dim - split_dim, input_dim - split_dim])
    return ConditionalAffineCoupling2(split_dim, hypernet)
