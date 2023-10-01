import gsd.hoomd
import freud
import numpy as np
from scipy.special import erf
from pathlib import Path
from natsort import natsorted
from scipy.fft import fft


PROJECT_DIR = Path().resolve()


def get_chebyshev_nodes(n, a, b):
    # Obtain Chebyshev points
    x = np.cos(np.pi * np.arange(n + 1) / n)
    # Rescale the points to the interval
    scaled_x = ((a + b) + ((b - a) * x)) * 0.5

    return scaled_x


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

    for frame in traj[:-1000]:
        msd_value = frame.log["hpmc/external/field/Harmonic/energy"][0]
        msd.append(msd_value)

    if spring == 0.0:
        print(np.mean(msd))

    return np.mean(msd) / spring


def main():
    reps = 15
    energies = []
    n_nodes = 31

    # FIXME: It seems that this integration scheme is overestimating
    # the free energy, the integration doesn't seem to be working

    for i in range(reps):
        data_path = PROJECT_DIR.joinpath("clenshaw_curtis", f"data_{i + 1}")
        path_list = natsorted(data_path.glob("**/trajectory.gsd"))
        # Define the integration limits
        lower_bound = 632.026
        # Lower bound is also the maximum of the lambda value
        # i.e. the spring constant
        upper_bound = 0.0
        # Get Chebyshev nodes
        scaled_nodes = get_chebyshev_nodes(n_nodes, lower_bound, upper_bound)
        result = 0.0
        n_particles = 256
        integrand = np.zeros_like(scaled_nodes)

        for i, (p, n) in enumerate(zip(path_list[1:], scaled_nodes[1:])):
            integrand[i] = compute_msd(p, n)

        # Now, we continue doing the Clenshaw-Curtis formula
        fx = integrand / (2.0 * n_nodes)
        # We need symmetric values for the FFT
        indices = np.concatenate((np.arange(n_nodes), np.arange(n_nodes, 0, -1)))
        g = np.real(fft(fx[indices]))
        # Now, compute the weights
        aug_g = np.concatenate(
            (
                np.array(g[0]).reshape(-1),
                g[1:n_nodes] + g[2 * (n_nodes + 1) : n_nodes : -1],
                np.array(g[n_nodes]).reshape(-1),
            )
        )
        w = np.zeros_like(aug_g)
        w[::2] = 2.0 / (1.0 - np.arange(0, n_nodes + 1, 2) ** 2)
        # Finally, rescale the weights to the integration interval
        w *= (upper_bound - lower_bound) * 0.5
        result = w @ aug_g
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
        A_0 = reference_energy(1.04086, lower_bound, n_particles)
        print(A_0)
        total_energy = result + A_0 + A_1
        energies.append(total_energy)
        print(total_energy)

    print(np.mean(energies), np.std(energies))


if __name__ == "__main__":
    main()
