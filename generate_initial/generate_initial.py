import random
import argparse
from particle import Particle
from writer import write_particles, set_initial_filename
from yaml_parser import parse_yaml

parser = argparse.ArgumentParser(description="Generate initial conditions for N-Body simulation")
parser.add_argument("-f", "--file", help="Yaml file to read initial conditions from", type=str)
parser.add_argument("-r", "--random", help="Generate random initial conditions with N particles", type=int)
parser.add_argument("-o", "--output", help="Output file", type=str, default="initial.ibin")

args = parser.parse_args()

print(args)

set_initial_filename(args.output)

if args.file:
    parse_yaml(args.file)

elif args.random:
    particles = []
    for i in range(args.random):
        particles.append(
            Particle((random.randint(0, 800), random.randint(0, 800)), (random.randint(-10, 10), random.randint(-10, 10)),
                     random.randint(0, 5)))

    write_particles(particles)

else:
    from pygame_editor import run_editor
    run_editor()

