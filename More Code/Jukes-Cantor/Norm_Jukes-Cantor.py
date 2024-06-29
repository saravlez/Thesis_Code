import math
import hyperiax
import jax
from Bio import Phylo
from io import StringIO
from functools import reduce
from jax import numpy as jnp
from jax.random import PRNGKey, split
from hyperiax.execution import LevelwiseTreeExecutor
from hyperiax.models import UpLambda
from hyperiax.mcmc import ParameterStore, UniformParameter
import matplotlib.pyplot as plt
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
    key = PRNGKey(42)

    from hyperiax.tree import HypTree, TreeNode
    from hyperiax.tree.childrenlist import ChildList

    # Random Example tree
    # newick_string = "(((A,C), (A,T)), ((C, T), (T, G)));"
    # newick = Phylo.read(StringIO(newick_string), "newick")
    # tree_newick = hyperiax.tree.builders.tree_from_newick(newick_string)
    tree_newick = hyperiax.tree.builders.symmetric_tree(h=5, degree=2)

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
            i.name = nucleotides[jnp.argmax(i['value'])]

    tree.plot_tree_text()

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
        'alpha': UniformParameter(value=.3, min=0, max=1/3),  # alpha parameter for kst (Jukes-Cantor)
    })

    # backwards filter. The operation is vmap'ed over the batch dimension (leading dimension)
    # @jax.jit
    def up(value, type, params, **args):
        def f(value, type):
            return {'value': transition(value, type, params)}
        return jax.vmap(f)(value, type)

    def norm_product_fuse(**kwargs):
        value = reduce(lambda x, y: x * y, kwargs['child_value']).reshape(-1)
        new_value = value / jnp.linalg.norm(value)
        log_sum = math.log(jnp.linalg.norm(value))
        return {'value': new_value, 'log_sum': log_sum}

    # create model and executor
    save_tree = tree.copy()
    upmodel = UpLambda(up_fn=up, fuse_fn=norm_product_fuse)
    upexec = LevelwiseTreeExecutor(upmodel)

    # execture backwards filter
    utree = upexec.up(tree, params.values())

    # It also normalizes the prior right now so
    utree.root.data['log_sum'] = 0
    utree.root.data['value'] = jnp.array([pi1, pi2, pi3, pi4])

    # print results
    total_log_sum = 0
    for node in utree.iter_bfs():
        print(node.name)
        print(node.data)
        total_log_sum += node.data['log_sum']

    # Compute Likelihood
    prior_vec = utree.root.data['value']
    h0_vec = utree.root.children[0].data['value']

    print(f"Before log : {jnp.dot(prior_vec, h0_vec)} and log sum: {total_log_sum}")
    log_lik = math.log(jnp.dot(prior_vec, h0_vec)) + total_log_sum
    print(log_lik)

    # Likelihood and gradients calculation over alpha range
    likelihood = []
    alpha_range = np.linspace(0.02, 0.32, 100)
    for new_alpha in alpha_range:
        # Update alpha
        params['alpha'].update(float(new_alpha), True)

        # execture backwards filter
        new_tree = upexec.up(save_tree, params.values())

        # Compute Log-Likelihood
        total_log_sum = 0
        for node in new_tree.iter_bfs():
            total_log_sum += node.data['log_sum']

        # Compute Likelihood
        prior_vec = new_tree.root.data['value']
        h0_vec = new_tree.root.children[0].data['value']

        new_lik = math.log(jnp.dot(prior_vec, h0_vec)) + total_log_sum
        likelihood.append(new_lik)

        print(f"Log-Likelihood with alpha {new_alpha}: {new_lik}")

    plt.plot(alpha_range, likelihood, label="Log-Likelihood")
    plt.xlabel('Alpha')
    plt.ylabel('Log-Likelihood')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
