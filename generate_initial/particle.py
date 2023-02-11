if __name__ == "__main__":
    print("This file is not meant to be run directly, please run generate_initial.py.")
    exit()

class Particle:
    def __init__(self, pos, vel, mass):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.radius = 2

    def __str__(self):
        return f"Particle({self.pos}, {self.vel}, {self.mass})"

    def __repr__(self):
        return str(self)