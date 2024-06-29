import math
from functools import reduce
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
import jax
import hyperiax
from Bio import Phylo
from jax import numpy as jnp
from hyperiax.models import UpLambda
from jax.random import PRNGKey
from hyperiax.execution import LevelwiseTreeExecutor
from hyperiax.mcmc import ParameterStore, UniformParameter

# create tree and initialize with noise
from hyperiax.tree import HypTree, TreeNode
from hyperiax.tree.childrenlist import ChildList


def kimura2_matrix(alpha, beta):
    """
    Generates the Kimura 2-Parameter transition matrix for given alpha and beta values.

    Assume:
     i) transitions (A <-> G and C <-> T) have rate alpha,
     ii) transversions (A <-> C, A <-> T, G <-> C, G <-> T) have rate beta.

    Input:
        alpha: The rate of transitions.
        beta: The rate of transversions.

    Output:
        matrix: Kimura 2-Parameter transition matrix.
    """

    # Conditions
    if alpha < 0 or beta < 0 or (alpha + 2*beta) > 1:
        raise ValueError("Alpha and Beta don't follow conditions")

    # 4x4 Transition matrix
    matrix = np.array([[1 - alpha - 2 * beta, alpha, beta, beta],
                      [alpha, 1 - alpha - 2 * beta, beta, beta],
                      [beta, beta, 1 - alpha - 2 * beta, alpha],
                      [beta, beta, alpha, 1 - alpha - 2 * beta]])

    # Probability Matrix rows check
    if not np.allclose(np.sum(matrix, axis=1), 1):
        raise ValueError("The rows are not equal to 1")

    return matrix


def main():
    key = PRNGKey(42)

    # Random Example tree
    # Initial Newick Tree
    newick_string = "(((A,C), (A,T)), ((C, T), (T, G)));"
    # newick = Phylo.read(StringIO(newick_string), "newick")
    tree_newick = hyperiax.tree.builders.tree_from_newick(newick_string)
    # tree_newick = hyperiax.tree.builders.symmetric_tree(h=5, degree=2)

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

    Phylo.draw(Phylo.read(StringIO(tree.to_newick()), "newick"), branch_labels=lambda c: c.branch_length)

    for i, node in enumerate(tree.iter_bfs(), start=-1):
        node.name = f'x_{i}' if node.name is None else f'{node.name}'

    # root, initial state prior
    pi1 = 0.9; pi2 = 0.08; pi3 = 0.01; pi4 = 0.01
    km10 = lambda params: jnp.diag([pi1, pi2, pi3, pi4])

    # inner node
    kst = lambda params: kimura2_matrix(params['alpha'], params['beta'])

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
        'alpha': UniformParameter(value=.05),  # alpha parameter for kst (Kimura)
        'beta': UniformParameter(value=.15),  # beta parameter for kst (Kimura)
    })

    # Save true params for later use
    true_params = ParameterStore({
        'alpha': UniformParameter(value=.05),
        'beta': UniformParameter(value=.15),
    })

    # backwards filter with normalized product
    def up(value, type, params, **args):
        return jax.vmap(lambda value, type: {'value': transition(value, type, params)})(value, type)

    def norm_product_fuse(**kwargs):
        value = reduce(lambda x, y: x * y, kwargs['child_value']).reshape(-1)
        new_value = value / jnp.linalg.norm(value)
        log_sum = math.log(jnp.linalg.norm(value))
        return {'value': new_value, 'log_sum': log_sum}
    # create model and executor
    upmodel = UpLambda(up_fn=up, fuse_fn=norm_product_fuse)
    upexec = LevelwiseTreeExecutor(upmodel)

    # execture backwards filter
    utree = upexec.up(tree, params.values())

    # It also normalizes the prior which I avoid
    utree.root.data['log_sum'] = 0
    utree.root.data['value'] = jnp.array([pi1, pi2, pi3, pi4])

    # print results
    total_log_sum = 0
    for node in utree.iter_bfs():
        # print(node.name)
        print(node.data)
        total_log_sum += node.data['log_sum']

    # Compute Log-Likelihood
    prior_vec = utree.root.data['value']
    h0_vec = utree.root.children[0].data['value']

    log_lik = math.log(jnp.dot(prior_vec, h0_vec)) + total_log_sum
    print(f"The log-likelihoog is {log_lik}")

    alpha_range = np.linspace(0.02, 0.32, 50)
    beta_range = np.linspace(0.02, 0.32, 50)
    likelihoods = np.zeros((len(alpha_range), len(beta_range)))

    for i, new_alpha in enumerate(alpha_range):
        for j, new_beta in enumerate(beta_range):
            # Update alpha
            params['alpha'].update(float(new_alpha), True)
            params['beta'].update(float(new_beta), True)

            # create model and executor
            upmodel = UpLambda(up_fn=up, fuse_fn=norm_product_fuse)
            upexec = LevelwiseTreeExecutor(upmodel)

            # execture backwards filter
            new_tree = upexec.up(tree, params.values())

            # It also normalizes the prior right now so
            new_tree.root.data['log_sum'] = 0

            # print results
            total_log_sum = 0
            for node in utree.iter_bfs():
                total_log_sum += node.data['log_sum']

            # Compute Log-Likelihood
            h0_vec = new_tree.root.children[0].data['value']
            new_lik = math.log(jnp.dot(prior_vec, h0_vec)) + total_log_sum
            likelihoods[i, j] = new_lik

    alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)

    plt.figure()
    levels = np.linspace(np.min(likelihoods), np.max(likelihoods), 75)
    cp = plt.contourf(alpha_grid, beta_grid, likelihoods.T, cmap='viridis', levels=levels)
    plt.colorbar(cp)

    # Plot the chosen parameters
    plt.scatter(true_params['alpha'].value, true_params['beta'].value, color='black', label='True Parameter')
    plt.legend()

    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Log-Likelihood Contour Plot')
    plt.savefig("K2_Contour.png")
    plt.show()


if __name__ == "__main__":
    main()
