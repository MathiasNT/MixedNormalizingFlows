import itertools
import pyro.distributions as dist
from pyro.nn import DenseNN
from torch import nn
import torch
from pyro.distributions.transforms import permute, batchnorm
from .ConditionalAffineCoupling2 import conditional_affine_coupling2
from ..nns import DropoutDenseNN


class ConditionalNormalizingFlowWrapper3(object):
    def __init__(self, transforms, flow, base_dist, condinet, batchnorms=None):
        self.dist = dist.ConditionalTransformedDistribution(base_dist, flow)
        self.condinet = condinet
        self.modules = nn.ModuleList(transforms).append(self.condinet)

        if batchnorms is not None:
            self.modules = self.modules.extend(batchnorms)

    def condition(self, context):
        rich_context = self.condinet(context)
        conditioned_dist = self.dist.condition(rich_context)
        return conditioned_dist

    def cuda(self):
        self.modules.cuda()


def conditional_normalizing_flow_factory3(flow_depth, problem_dim, c_net_depth, c_net_h_dim, context_dim,
                                          context_n_h_dim, context_n_depth, rich_context_dim, batchnorm_momentum, cuda,
                                          coupling_dropout=None, context_dropout=None):
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


    # Permutes are needed to be able to transform all dimensions.
    # Note that the transform is fixed here since we only have 2 dimensions. For more dimensions don't fix it and let it be random.
    perms = [permute(input_dim=problem_dim, permutation=torch.tensor([1, 0])) for i in range(flow_depth)]

    # If we want batchnorm add those in. Then sandwich the steps together to a flow
    if batchnorm_momentum is None:
        batchnorms = None
        flows = list(itertools.chain(*zip(transforms, perms)))[:-1]
    else:
        batchnorms = [batchnorm(input_dim=problem_dim, momentum=batchnorm_momentum) for i in range(flow_depth)]
        for bn in batchnorms:
            bn.gamma.data += torch.ones(problem_dim)
        flows = list(itertools.chain(*zip(batchnorms, transforms, perms)))[1:-1]

    # We define the conditioning network
    context_hidden_dims = [context_n_h_dim for i in range(context_n_depth)]
    if context_dropout is None:
        condinet = DenseNN(input_dim=context_dim, hidden_dims=context_hidden_dims, param_dims=[rich_context_dim])
    else:
        condinet = DropoutDenseNN(input_dim=context_dim, hidden_dims=context_hidden_dims, param_dims=[rich_context_dim],
                                  dropout=context_dropout)
    # We define the normalizing flow wrapper
    normalizing_flow = ConditionalNormalizingFlowWrapper3(transforms, flows, base_dist, condinet, batchnorms)
    if cuda:
        normalizing_flow.cuda()

    return normalizing_flow
