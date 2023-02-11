import struct

if __name__ == "__main__":
    print("This file is not meant to be run directly, please run generate_initial.py.")
    exit()

initial_filename = "initial.ibin"


def set_initial_filename(filename):
    global initial_filename
    initial_filename = filename


def write_particles(particles, threads_per_block=1024, dt=0.01, epochs=1000, gravity_constant=1):
    with open(initial_filename, "wb") as f:
        # Write header

        f.write("IBIN".encode("utf-8"))
        f.write(struct.pack("f", dt))
        f.write(struct.pack("i", threads_per_block))
        f.write(struct.pack("q", epochs))
        f.write(struct.pack("f", gravity_constant))
        f.write(struct.pack("i", len(particles)))

        f.seek(4, 1)

        # Write particles

        for particle in particles:
            f.write(struct.pack("fffff", float(particle.pos[0]), float(particle.pos[1]), float(particle.vel[0]),
                                float(particle.vel[1]), float(particle.mass)))

    print(f"Saved to {initial_filename}")
