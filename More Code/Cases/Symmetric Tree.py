import math
from io import StringIO
from Bio import Phylo
from jax.random import PRNGKey, split
import hyperiax
import jax
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
import numpy as np


def jukes_cantor_matrix(alpha):
    """
    Generates the Jukes-Cantor matrix for a given alpha value.

    Assume
     i) each base in the sequence has an equal probability of being substituted, and
     ii) if a nucleotide substitution occurs, all other nucleotides have the same probability to replace it

    Input:
        alpha: The probability of substitution

    Output:
        matrix: Jukes-Cantor transition matrix.
    """
    # Conditions
    if alpha < 0 or alpha >= (1 / 3):
        raise ValueError("Alpha must be a fraction between 0 and 1/3")

    # 4x4 Transition matrix
    matrix = np.array([[1 - 3 * alpha, alpha, alpha, alpha],
                      [alpha, 1 - 3 * alpha, alpha, alpha],
                      [alpha, alpha, 1 - 3 * alpha, alpha],
                      [alpha, alpha, alpha, 1 - 3 * alpha]])

    # Probability Matrix rows check
    if np.all((np.sum(matrix, axis=1)) != 1):
        raise ValueError("The rows are not equal to 1")

    # print(f"kernel :\n{matrix}")
    return matrix


def main():
    # Example with Jukes-Cantor Kernel
    key = PRNGKey(42)

    # Initial random newick tree
    newick_string = "(((A,C), (A,T)), ((C, T), (T, G)));"
    tree_newick = hyperiax.tree.builders.tree_from_newick(newick_string)

    # Or symmetric tree with random observations
    # tree_newick = hyperiax.tree.builders.symmetric_tree(h=4, degree=2)

    # Prior Data
    prior = TreeNode(); prior.name = "prior"
    # Root Data
    x0 = tree_newick.root; x0.name = "x0"
    x0.parent = prior; prior.children = ChildList([x0])
    x0.children = tree_newick.root.children

    tree = HypTree(prior)
    print('Tree:', tree)

    # set types to select the right transitions
    # types
    troot = 0; tinner_node = 1; tleaf_node = 2

    for i in tree.iter_bfs():
        i['type'] = tinner_node
        i['log_sum'] = 0

    for i in tree.iter_leaves():
        i['type'] = tleaf_node

    tree.root['type'] = troot
    x0['type'] = troot

    # number of states
    R = 4

    # root value
    tree.root['value'] = jnp.zeros(R)

    # observations
    nucleotides = ['A', 'G', 'C', 'T']
    for i in tree.iter_leaves():
        if i.name in nucleotides:
            i['value'] = jnp.eye(R)[nucleotides.index(i.name)]
        else:
            i['value'] = jnp.eye(R)[np.random.choice(4)]
            i.name = nucleotides[np.argmax(i['value'])]

    Phylo.draw(Phylo.read(StringIO(tree.to_newick()), "newick"), branch_labels=lambda c: c.branch_length)

    for i, node in enumerate(tree.iter_bfs(), start=-1):
        if node.name is None:
            node.name = f'x_{i}' if node.name is None else f'{node.name}'

    # root, initial state prior
    pi1 = 0.9; pi2 = 0.08; pi3 = 0.01; pi4 = 0.01
    km10 = lambda params: jnp.diag([pi1, pi2, pi3, pi4])

    # inner node
    kst = lambda params: jukes_cantor_matrix(params['alpha'])

    # leaves
    lambdi = lambda params: jnp.eye(R)

    # using jax.lax.cond instead of python ifs
    def transition(value, type, params):
        return jax.lax.cond(type == tinner_node,
                            lambda: jnp.dot(kst(params), value),
                            lambda: jax.lax.cond(type == tleaf_node,
                                                 lambda: jnp.dot(kst(params), jnp.dot(lambdi(params), value)),
                                                 lambda: jnp.array([pi1, pi2, pi3, pi4])
                                                 )
                            )

    # parameters, alpha with uniform prior
    params = ParameterStore({
        'alpha': UniformParameter(value=.1, min=0, max=1/3),  # alpha parameter for kst (Jukes-Cantor)
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

    # It also normalizes the prior right now so
    utree.root.data['log_sum'] = 0
    utree.root.data['value'] = jnp.array([pi1, pi2, pi3, pi4])

    # print results
    total_log_sum = 0
    for node in utree.iter_bfs():
        print(node.data)
        total_log_sum += node.data['log_sum']

    # Compute Log-Likelihood
    prior_vec = utree.root.data['value']
    h0_vec = utree.root.children[0].data['value']

    log_lik = math.log(jnp.dot(prior_vec, h0_vec)) + total_log_sum
    print(f"The log-likelihoog is {log_lik}")


if __name__ == "__main__":
    main()
