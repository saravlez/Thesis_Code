import hyperiax
import jax
import math
from jax.random import PRNGKey, split
from jax import numpy as jnp
from hyperiax.execution import LevelwiseTreeExecutor
from hyperiax.models import UpLambda
from hyperiax.models.functional import product_fuse_children
from hyperiax.mcmc import ParameterStore, UniformParameter
from hyperiax.mcmc.metropolis_hastings import metropolis_hastings
from hyperiax.mcmc.plotting import trace_plots

# create tree and initialize with noise
from hyperiax.tree import HypTree, TreeNode
from hyperiax.tree.childrenlist import ChildList


def main():
    key = PRNGKey(42)

    # example tree, see Figure 1 / Figure 4 tree in https://arxiv.org/abs/2203.04155
    root = TreeNode();  # x_{-1}
    x0 = TreeNode(); x0.parent = root; root.children = ChildList([x0])
    x1 = TreeNode(); x1.parent = x0;
    x3 = TreeNode(); x3.parent = x0;
    x0.children = ChildList([x1, x3])
    x2 = TreeNode(); x2.parent = x1; x1.children = ChildList([x2])
    v3 = TreeNode(); v3.parent = x2; x2.children = ChildList([v3])
    x4 = TreeNode(); x4.parent = x3;
    v2 = TreeNode(); v2.parent = x3; x3.children = ChildList([x4, v2])
    v1 = TreeNode(); v1.parent = x4; x4.children = ChildList([v1])
    v1.children = v2.children = v3.children = ChildList()

    tree = HypTree(root)
    print('Tree:', tree)
    tree.plot_tree_text()

    # types for correct transitions
    troot = 0;
    tinner_node = 1;
    tleaf_node = 2;
    tree.root['type'] = troot
    x1['type'] = x2['type'] = x3['type'] = x4['type'] = tinner_node
    v1['type'] = v2['type'] = v3['type'] = tleaf_node

    # number of states
    R = 3

    # root value
    tree.root['value'] = jnp.zeros(R)
    x0['type'] = troot

    # observations
    v1['value'] = jnp.eye(R)[0]
    v2['value'] = jnp.eye(R)[1]
    v3['value'] = jnp.eye(R)[2]

    # initial state prior
    pi1 = pi2 = pi3 = 1 / 3;
    km10 = lambda params: jnp.diag([pi1, pi2, pi3])
    # inner node
    kst = lambda params: jnp.array([[1. - params['theta'], params['theta'], 0.],
                                    [.25, .5, .25],
                                    [.4, .3, .3]])
    # leaves
    lambdi = lambda params: jnp.array([[1., 1., 0.],
                                       [1., 1., 0.],
                                       [0., 0., 1.]])

    # transitions inside the tree
    def transition(value, type, params):
        return jax.lax.cond(type == tinner_node,
                            lambda: jnp.dot(kst(params), value),
                            lambda: jax.lax.cond(type == tleaf_node,
                                                 lambda: jnp.dot(lambdi(params), value),
                                                 lambda: jnp.array([pi1, pi2, pi3])
                                                 )
                            )

    # parameters, theta with uniform prior
    params = ParameterStore({
        'theta': UniformParameter(.5),  # theta parameter for kst
    })

    # backwards filter. The operation is vmap'ed over the batch dimension (leading dimension)
    # @jax.jit
    def up(value, type, params, **args):
        return jax.vmap(lambda value, type: {'value': transition(value, type, params)})(value, type)

    # create model and executor
    upmodel = UpLambda(up_fn=up, fuse_fn=product_fuse_children(axis=0))
    upexec = LevelwiseTreeExecutor(upmodel)

    # execture backwards filter
    utree = upexec.up(tree, params.values())

    # print results
    for node in utree.iter_bfs():
        print(node.data)

    prior_vec = jnp.array([pi1, pi2, pi3])
    h0_vec = utree.root.children[0].data['value']

    print(f"Before log : {jnp.dot(prior_vec, h0_vec)}")
    log_lik = math.log(jnp.dot(prior_vec, h0_vec))
    print(log_lik)


if __name__ == "__main__":
    main()
