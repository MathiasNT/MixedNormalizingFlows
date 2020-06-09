import torch.distributions as dist
from torch import nn
import torch
from pyro.distributions.transforms import affine_coupling, permute, batchnorm

import itertools


class NormalizingFlowWrapper(object):
    def __init__(self, transforms, flow, base_dist, batchnorms=None):
        self.dist = dist.TransformedDistribution(base_dist, flow)
        self.modules = nn.ModuleList(transforms)

        if batchnorms is not None:
            self.modules = self.modules.extend(batchnorms)

    def cuda(self):
        self.modules.cuda()


def normalizing_flow_factory(flow_depth, problem_dim, c_net_depth, c_net_h_dim, batchnorm_momentum, cuda):
    # We define the base distribution
    if cuda:
        base_dist = dist.Normal(torch.zeros(problem_dim).cuda(), torch.ones(problem_dim).cuda())
    else:
        base_dist = dist.Normal(torch.zeros(problem_dim), torch.ones(problem_dim))

    # We define the transformations
    transforms = [affine_coupling(input_dim=problem_dim,
                                  hidden_dims=[c_net_h_dim for i in range(c_net_depth)]) for i in range(flow_depth)]

    # We need to permute dimensions to affect them both THIS NEEDS A FIX
    perms = [permute(2, torch.tensor([1, 0])) for i in range(flow_depth)]

    # If we want batchnorm add those in. Then sandwich the steps together to a flow
    if batchnorm_momentum is None:
        batchnorms = None
        flows = list(itertools.chain(*zip(transforms, perms)))[:-1]
    else:
        batchnorms = [batchnorm(input_dim=problem_dim, momentum=batchnorm_momentum) for i in range(flow_depth)]
        for bn in batchnorms:
            bn.gamma.data += torch.ones(problem_dim)
        flows = list(itertools.chain(*zip(batchnorms, transforms, perms)))[1:-1]


    # We define the normalizing flow wrapper
    normalizing_flow = NormalizingFlowWrapper(transforms, flows, base_dist, batchnorms)
    if cuda:
        normalizing_flow.cuda()
    return normalizing_flow


def planar_normalizing_flow_factory(flow_depth, problem_dim, c_net_depth, c_net_h_dim, batchnorm_momentum, cuda):
    # We define the base distribution
    if cuda:
        base_dist = dist.Normal(torch.zeros(problem_dim).cuda(), torch.ones(problem_dim).cuda())
    else:
        base_dist = dist.Normal(torch.zeros(problem_dim), torch.ones(problem_dim))

    # We define the transformations
    transforms = [affine_coupling(input_dim=problem_dim,
                                  hidden_dims=[c_net_h_dim for i in range(c_net_depth)]) for i in range(flow_depth)]

    # We need to permute dimensions to affect them both THIS NEEDS A FIX
    perms = [permute(2, torch.tensor([1, 0])) for i in range(flow_depth)]

    # If we want batchnorm add those in. Then sandwich the steps together to a flow
    if batchnorm_momentum is None:
        batchnorms = None
        flows = list(itertools.chain(*zip(transforms, perms)))[:-1]
    else:
        batchnorms = [batchnorm(input_dim=problem_dim, momentum=batchnorm_momentum) for i in range(flow_depth)]
        for bn in batchnorms:
            bn.gamma.data += torch.ones(problem_dim)
        flows = list(itertools.chain(*zip(batchnorms, transforms, perms)))[1:-1]


    # We define the normalizing flow wrapper
    normalizing_flow = NormalizingFlowWrapper(transforms, flows, base_dist, batchnorms)
    if cuda:
        normalizing_flow.cuda()
    return normalizing_flow

