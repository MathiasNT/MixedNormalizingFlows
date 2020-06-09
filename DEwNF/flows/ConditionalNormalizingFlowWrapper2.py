import itertools
import pyro.distributions as dist
from pyro.nn import DenseNN
from torch import nn
import torch
from pyro.distributions.transforms import permute
from .ConditionalAffineCoupling2 import conditional_affine_coupling2
from ..nns import DropoutDenseNN


class ConditionalNormalizingFlowWrapper2(object):
    def __init__(self, transforms, flow, base_dist, condinet):
        self.dist = dist.ConditionalTransformedDistribution(base_dist, flow)
        self.condinet = condinet
        self.modules = nn.ModuleList(transforms).append(self.condinet)

    def condition(self, context):
        rich_context = self.condinet(context)
        conditioned_dist = self.dist.condition(rich_context)
        return conditioned_dist

    def cuda(self):
        self.modules.cuda()


def conditional_normalizing_flow_factory2(flow_depth, problem_dim, c_net_depth, c_net_h_dim, context_dim,
                                          context_n_h_dim, context_n_depth, rich_context_dim, cuda,
                                          coupling_dropout=None, context_dropout=None):
    # We define the base distribution
    if cuda:
        base_dist = dist.Normal(torch.zeros(problem_dim).cuda(), torch.ones(problem_dim).cuda())
    else:
        base_dist = dist.Normal(torch.zeros(problem_dim), torch.ones(problem_dim))

    # We define the transformations
    transforms = [conditional_affine_coupling2(input_dim=problem_dim,
                                               context_dim=context_dim,
                                               hidden_dims=[c_net_h_dim for i in range(c_net_depth)], # Note array here to create multiple layers in DenseNN
                                               rich_context_dim=rich_context_dim,
                                               dropout=coupling_dropout)
                  for i in range(flow_depth)]


    # need a fix for this
    perms = [permute(2, torch.tensor([1, 0])) for i in range(flow_depth)]

    # We sandwich the AffineCouplings and permutes together. Unelegant hotfix to remove last permute but it works
    flows = list(itertools.chain(*zip(transforms, perms)))[:-1]

    # We define the conditioning network
    context_hidden_dims = [context_n_h_dim for i in range(context_n_depth)]
    if context_dropout is None:
        condinet = DenseNN(input_dim=context_dim, hidden_dims=context_hidden_dims, param_dims=[rich_context_dim])
    else:
        condinet = DropoutDenseNN(input_dim=context_dim, hidden_dims=context_hidden_dims, param_dims=[rich_context_dim],
                                  dropout=context_dropout)
    # We define the normalizing flow wrapper
    normalizing_flow = ConditionalNormalizingFlowWrapper2(transforms, flows, base_dist, condinet)
    if cuda:
        normalizing_flow.cuda()

    return normalizing_flow
