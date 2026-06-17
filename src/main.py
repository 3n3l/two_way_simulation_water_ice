import sys, os, math

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from src.presets import ice_presets, water_presets, mixed_presets
from src.parsers.parsing import parser, add_configuration
from src.simulation import GGUI_Simulation, GUI_Simulation
from src.samplers import PoissonDiskSampler

from two_way_simulation import TwoWay_MLSMPM

import taichi as ti


def main():
    configurations = ice_presets + water_presets + mixed_presets
    add_configuration(configurations)
    arguments = parser.parse_args()
    print(parser.epilog)

    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=arguments.debug, verbose=arguments.verbose, unrolling_limit=0)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug, verbose=arguments.verbose, unrolling_limit=0)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug, verbose=arguments.verbose, unrolling_limit=0)

    initial_configuration = arguments.configuration % len(configurations)
    name = "Two-Way Simulation of Water & Ice"
    prefix = "TWS_MLSMPM"

    d = arguments.dimension
    q = 2**arguments.quality
    
    n_grid = math.ceil(128 * q)
    dx = 1 / n_grid
    n_particles_cell = 4
    radius = dx / (2 * (n_particles_cell ** (1 / 3)))
    vol_0 = (0.5 * dx) ** 3

    # Make a rough guess of maximum possible amount of particles from volumes:
    max_volume = 0.0
    for configuration in configurations:
        if (volume := configuration.volume()) > max_volume:
            max_volume = volume
    max_particles = math.ceil(max_volume / (vol_0))

    solver = TwoWay_MLSMPM(max_particles=max_particles, n_dimensions=d, n_grid=n_grid, vol_0=vol_0)
    poisson_disk_sampler = PoissonDiskSampler(solver=solver, r=radius*1.5, k=30)

    if arguments.gui.lower() == "ggui":
        simulation = GGUI_Simulation(
            initial_configuration=initial_configuration,
            configurations=configurations,
            sampler=poisson_disk_sampler,
            res=(720, 720),
            prefix=prefix,
            solver=solver,
            radius=radius,
            name=name,
            quality=q,
        )
        simulation.run()
    elif arguments.gui.lower() == "gui":
        simulation = GUI_Simulation(
            initial_configuration=initial_configuration,
            configurations=configurations,
            sampler=poisson_disk_sampler,
            prefix=prefix,
            solver=solver,
            radius=radius,
            name=name,
            quality=q,
            res=720,
        )
        simulation.run()

    print("\n", "#" * 100, sep="")
    print("###", name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
