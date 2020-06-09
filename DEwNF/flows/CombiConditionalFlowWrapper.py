import itertools
import pyro.distributions as dist
from pyro.nn import DenseNN
from torch import nn
import torch
from pyro.distributions.transforms import permute, batchnorm

from ..nns import DropoutDenseNN
from .ConditionalAffineCoupling2 import conditional_affine_coupling2
from DEwNF.flows.InvertedConditionalPlanar import inverted_conditional_planar
from DEwNF.flows.ConditionalNormalizingFlowWrapper3 import ConditionalNormalizingFlowWrapper3
from DEwNF.flows import InvertedConditionalPlanar, ConditionalAffineCoupling2


def combi_conditional_normalizing_flow_factory(flow_depth, problem_dim, c_net_depth, c_net_h_dim, context_dim,
                                               context_n_h_dim, context_n_depth, rich_context_dim, batchnorm_momentum,
                                               cuda, coupling_dropout=None, context_dropout=None, planar_first=True):
    assert (flow_depth % 2 == 0), "The flow depth must be divisible by 2 to allow both Planar and AC transforms"

    if cuda:
        base_dist = dist.Normal(torch.zeros(problem_dim).cuda(), torch.ones(problem_dim).cuda())
    else:
        base_dist = dist.Normal(torch.zeros(problem_dim), torch.ones(problem_dim))

    # We define the transformations
    affine_couplings = [conditional_affine_coupling2(input_dim=problem_dim,
                                                     context_dim=context_dim,
                                                     hidden_dims=[c_net_h_dim for i in range(c_net_depth)],
                                                     rich_context_dim=rich_context_dim,
                                                     dropout=coupling_dropout)
                        for i in range(flow_depth // 2)]

    planars = [inverted_conditional_planar(input_dim=problem_dim,
                                           context_dim=rich_context_dim,
                                           hidden_dims=[c_net_h_dim for i in range(c_net_depth)],
                                           )for i in range(flow_depth // 2)]
    transforms = affine_couplings + planars

    # Permutes are needed to be able to transform all dimensions.
    # Note that the transform is fixed here since we only have 2 dimensions. For more dimensions let it be random.
    perms = [permute(input_dim=problem_dim, permutation=torch.tensor([1, 0])) for i in range(flow_depth//2)]

    # Assemble the flow
    if planar_first:
        flows = list(itertools.chain(*zip(planars, affine_couplings, perms)))[:-1]
    else:
        flows = list(itertools.chain(*zip(affine_couplings, planars, perms)))[:-1]

    # If we want batchnorm add those in. Then sandwich the steps together to a flow
    if batchnorm_momentum is None:
        batchnorms = None
    else:
        bn_flow = flows[:1]
        batchnorms = []
        for trans in flows[1:]:
            if isinstance(trans, ConditionalAffineCoupling2) or isinstance(trans, InvertedConditionalPlanar):
                batchnorms.append(batchnorm(input_dim=problem_dim, momentum=batchnorm_momentum))
                bn_flow.append(batchnorms[-1])
            bn_flow.append(trans)
        flows = bn_flow
        for bn in batchnorms:
            bn.gamma.data += torch.ones(problem_dim)

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
