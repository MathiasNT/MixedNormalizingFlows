import torch
import pyro.distributions as dist
from pyro.distributions.transforms import permute, batchnorm
from pyro.nn import DenseNN
import itertools

from DEwNF.flows.ConditionalNormalizingFlowWrapper3 import ConditionalNormalizingFlowWrapper3
from DEwNF.flows.InvertedConditionalPlanar import inverted_conditional_planar
from ..nns import DropoutDenseNN


def inverted_conditional_planar_flow_factory(flow_depth, problem_dim, c_net_depth, c_net_h_dim, context_dim,
                                             context_n_h_dim, context_n_depth, rich_context_dim, batchnorm_momentum,
                                             cuda, context_dropout=None):
    if cuda:
        base_dist = dist.Normal(torch.zeros(problem_dim).cuda(), torch.ones(problem_dim).cuda())
    else:
        base_dist = dist.Normal(torch.zeros(problem_dim), torch.ones(problem_dim))

    # We define the transformations
    transforms = [inverted_conditional_planar(input_dim=problem_dim,
                                              context_dim=rich_context_dim,
                                              hidden_dims=[c_net_h_dim for i in range(c_net_depth)],
                                              )for i in range(flow_depth)]

    # If we want batchnorm add those in. Then sandwich the steps together to a flow
    if batchnorm_momentum is None:
        batchnorms = None
        flows = transforms
    else:
        batchnorms = [batchnorm(input_dim=problem_dim, momentum=batchnorm_momentum) for i in range(flow_depth)]
        for bn in batchnorms:
            bn.gamma.data += torch.ones(problem_dim)
        flows = list(itertools.chain(*zip(batchnorms, transforms)))[1:]


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
