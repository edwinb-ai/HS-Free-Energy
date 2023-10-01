import numpy as np
from constants import kronrod_weights, gauss_kronrod_nodes


def get_kronrod_weight(nodes_count):
    available_kronrod_weights = np.append(
        kronrod_weights[nodes_count], kronrod_weights[nodes_count][:-1][::-1]
    )

    return available_kronrod_weights


def get_gauss_kronrod_nodes(nodes_count):
    available_gauss_kronrod_nodes = np.append(
        gauss_kronrod_nodes[nodes_count],
        np.negative(gauss_kronrod_nodes[nodes_count][:-1][::-1]),
    )

    return available_gauss_kronrod_nodes


def main():
    # This number should be the number of tasks and also
    # quadrature nodes to use
    n_tasks = 61
    # Define the integration limits
    lower_bound = 1000.0
    # Lower bound is also the maximum of the lambda value
    # i.e. the spring constant
    upper_bound = 0.0
    # Obtain the nodes and weights
    nodes = get_gauss_kronrod_nodes(n_tasks)
    # Scale the nodes
    mid = 0.5 * (upper_bound + lower_bound)
    dx = 0.5 * (upper_bound - lower_bound)
    scaled_nodes = mid + (nodes * dx)
    array_id = np.arange(1, n_tasks + 1, step=1)
    data = np.c_[array_id, scaled_nodes]
    np.savetxt(
        "config.txt",
        data,
        fmt=["%.d", "%.14f"],
        delimiter="\t",
        header="ArrayTaskID\tSpring",
        comments="",
    )


if __name__ == "__main__":
    main()
