import gsd.hoomd
import freud
import numpy as np
from constants import kronrod_weights, gauss_kronrod_nodes
from scipy.special import erf
from pathlib import Path
from natsort import natsorted


PROJECT_DIR = Path().resolve()


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


def overlap(a, lamb, sigma=1.0):
    term_1 = erf(np.sqrt(lamb / 2.0) * (sigma + a))
    term_2 = erf(np.sqrt(lamb / 2.0) * (sigma - a))
    term_3 = np.exp(-(lamb / 2.0) * (sigma - a) ** 2)
    term_4 = np.exp(-(lamb / 2.0) * (sigma + a) ** 2)

    acc_1 = (term_1 + term_2) / 2.0
    acc_2 = (term_3 - term_4) / (np.sqrt(2.0 * np.pi * lamb) * a)

    return acc_1 - acc_2


def reference_energy(density, lamb, npart):
    term_1 = np.log(density) / npart
    term_2 = 3.0 * (1.0 - (1.0 / npart)) * np.log(lamb / np.pi) / 2.0

    return term_1 + term_2


def compute_distance(path):
    traj = gsd.hoomd.open(path, "r")
    frame = traj[0]
    # Compute average distance
    positions = frame.particles.position
    box = frame.configuration.box
    aq = freud.locality.AABBQuery.from_system((box, positions))
    distances = []
    for bond in aq.query(positions, dict(num_neighbors=12, exclude_ii=True)):
        distances.append(bond[2])

    return np.mean(distances)


def compute_msd(path, spring):
    traj = gsd.hoomd.open(path, "r")
    msd = []

    for frame in traj[:-2000]:
        msd_value = frame.log["hpmc/external/field/Harmonic/energy_translational"][0]
        msd.append(msd_value)

    return np.mean(msd) / spring


def main():
    reps = 6
    energies = []
    n_particles = 500
    density = 1.04086
    # Define the integration limits
    lower_bound = 1000.0
    # Lower bound is also the maximum of the lambda value
    # i.e. the spring constant
    upper_bound = 0.0
    # Scale the nodes
    mid = 0.5 * (upper_bound + lower_bound)
    dx = 0.5 * (upper_bound - lower_bound)
    # Obtain the nodes and weights
    nodes = get_gauss_kronrod_nodes(61)
    weights = get_kronrod_weight(61)
    scaled_nodes = mid + (nodes * dx)

    for i in range(reps):
        data_path = PROJECT_DIR.joinpath(f"Npart={n_particles}_moreq", f"data_{i + 1}")
        path_list = natsorted(data_path.glob("**/trajectory.gsd"))
        result = 0.0

        for p, w, n in zip(path_list, weights, scaled_nodes):
            integral = compute_msd(p, n)
            result += w * dx * integral

        result /= n_particles

        print(result)

        distance = compute_distance(path_list[-1])
        # This is the overlap probability distribution as described in Frenkel & Ladd
        # (see appendix of the original paper)
        # The number `n = 12` is the total neighbors for the FCC crystal
        corrections = (
            -np.log(1.0 - overlap(np.mean(distance), lower_bound)) * 12.0 / 2.0
        )
        # There is an additional term that is needed, because of finite-size errors
        finite_correction = 1e-3 * lower_bound * np.exp(-lower_bound / 1e2)
        A_1 = corrections + finite_correction
        print(A_1)
        A_0 = reference_energy(density, lower_bound, n_particles)
        print(A_0)
        total_energy = result + A_0 + A_1
        energies.append(total_energy)
        print(total_energy)

    print(np.mean(energies), np.std(energies))


if __name__ == "__main__":
    main()
