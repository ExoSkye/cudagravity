import ruamel.yaml

from writer import write_particles
from particle import Particle


def parse_yaml(filename):
    with open(filename, "r") as f:
        yaml = ruamel.yaml.load(f, Loader=ruamel.yaml.RoundTripLoader)

    threads = yaml["threads"]
    dt = yaml["dt"]
    epochs = yaml["epochs"]
    gravity_constant = yaml["gravity_constant"]

    if gravity_constant == "real":
        gravity_constant = 6.67408e-11

    particles = []

    for particle in yaml["particles"]:
        particles.append(
            Particle(
                (particle["position"][0], particle["position"][1]),
                (particle["velocity"][0], particle["velocity"][1]),
                particle["mass"]
            )
        )

    write_particles(particles, threads, dt, epochs, gravity_constant)
