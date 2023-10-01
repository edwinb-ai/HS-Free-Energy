import freud
import hoomd
import argparse
import gsd.hoomd
import numpy as np
from pathlib import Path


PROJECT_DIR = Path().resolve()


def simulation(spring_constant, rep):
    # Create the initial lattice, in this case, FCC
    fcc = freud.data.UnitCell.fcc()
    box, positions = fcc.generate_system(num_replicas=4)
    N_particles = positions.shape[0]
    # Define the density to work with and compute the size of the box
    low_density = 1.04086
    low_L = np.cbrt(N_particles / low_density)
    # Re-scale the positions of the particles to the new box
    positions *= [low_L, low_L, low_L] / box.L
    # Define here the reference positions and orientations for the
    # harmonic potential implementation
    reference_positions = np.copy(positions)
    reference_orientations = [(1, 0, 0, 0)] * N_particles
    # Then we deal with the initialization of the system for the simulation
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = positions
    frame.particles.orientation = np.copy(reference_orientations)
    # There are two types of particles: mobile and immobile, the immobile
    # particle is the reference particle (carrier according to Vega & Noya)
    frame.particles.types = ["hs", "hs_no_motion"]
    frame.particles.typeid = np.zeros((N_particles,))
    # This is the ID for the carrier particle
    frame.particles.typeid[0] = 1
    frame.configuration.box = [low_L, low_L, low_L, 0, 0, 0]

    # Create a directory with the information of this state point
    data_path = PROJECT_DIR.joinpath(
        f"Npart={N_particles}",
        f"data_{rep}",
        f"k={spring_constant:.6g}_npart={N_particles}",
    )
    data_path.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=data_path.joinpath("lattice.gsd"), mode="w") as f:
        f.append(frame)

    # We start by defining the simulation object
    cpu = hoomd.device.CPU()
    rng = np.random.default_rng()
    sim = hoomd.Simulation(device=cpu, seed=rng.integers(2**16))
    sim.create_state_from_snapshot(frame)
    # We define the integrator
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.01)
    mc.shape["hs"] = dict(diameter=1.0)
    mc.shape["hs_no_motion"] = mc.shape["hs"]
    # Set movement to zero for the single particle
    mc.a["hs_no_motion"] = 0.0
    mc.d["hs_no_motion"] = 0.0
    sim.operations.integrator = mc

    # During equilibration, we need the harmonic constraints activated
    harmonic = hoomd.hpmc.external.field.Harmonic(
        reference_positions,
        reference_orientations,
        k_translational=spring_constant,
        k_rotational=1.0,
        symmetries=[[1, 0, 0, 0]],
    )
    mc.external_potential = harmonic

    # Add logging
    logger = hoomd.logging.Logger()
    logger.add(harmonic, quantities=["energy_translational"])
    gsd_writer = hoomd.write.GSD(
        filename=data_path.joinpath("trajectory.gsd"),
        trigger=hoomd.trigger.And(
            [hoomd.trigger.Periodic(1_000), hoomd.trigger.After(200_000)]
        ),
        mode="wb",
        filter=hoomd.filter.All(),
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    # * Equilibrate the system
    # Add a displacement tuner with a target acceptance ratio of 0.4
    tune = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["d"],
        target=0.4,
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(sim.timestep + 20_000),
            ]
        ),
    )
    sim.operations.tuners.append(tune)
    # Do a quick run to adjust the move size
    sim.run(2e4)
    # Always remove the tuner
    sim.operations.tuners.remove(tune)
    print("Starting equilibration run...")
    sim.run(1e6)


def main():
    # Run the argparse
    parser = argparse.ArgumentParser(description="Collect state point.")
    parser.add_argument("--spring", type=float)
    parsed = parser.parse_args()
    lamb = parsed.spring

    for i in range(15):
        # The actual value of the constant should be doubled, this is because
        # of the definition of the harmonic force that acts on the particles
        simulation(lamb * 2.0, i + 1)


if __name__ == "__main__":
    main()
