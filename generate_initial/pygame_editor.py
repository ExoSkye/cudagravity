import pygame
from particle import Particle
from writer import write_particles

if __name__ == "__main__":
    print("This file is not meant to be run directly, please run generate_initial.py.")
    exit()

pygame.init()


def run_editor():
    screen = pygame.display.set_mode((800, 800))

    particles = []

    state = 0

    # States:
    # 0: Waiting for first click
    # 1: Drag for velocity, 0 for no velocity, ESC for no particle

    current_particle = Particle((0, 0), (0, 0), 0)
    current_line_end = (0, 0)

    mouse_down_handled = False

    while True:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if state == 0 and not mouse_down_handled:
                    mouse_down_handled = True
                    state = 1
                    current_particle.pos = event.pos

                if state == 1 and not mouse_down_handled:
                    mouse_down_handled = True
                    current_particle.vel = (
                        (event.pos[0] - current_particle.pos[0]) / 100, (event.pos[1] - current_particle.pos[1]) / 100)
                    particles.append(current_particle)
                    current_particle = Particle((0, 0), (0, 0), 0)
                    state = 0
                    print(particles)

            elif event.type == pygame.MOUSEMOTION:
                if state == 1:
                    current_line_end = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down_handled = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state = 0
                    current_particle = Particle((0, 0), (0, 0), 0)
                    current_line_end = (0, 0)

                elif event.key == pygame.K_0:
                    current_particle.vel = (0, 0)

                elif event.key == pygame.K_KP_PLUS:
                    current_particle.mass += 1
                    current_particle.radius += 1

                elif event.key == pygame.K_KP_MINUS:
                    current_particle.mass -= 1
                    current_particle.radius -= 1

                elif event.key == pygame.K_RETURN:
                    write_particles(particles)

        if state == 1:
            pygame.draw.line(screen, (255, 255, 255), current_particle.pos, current_line_end, 1)

        for particle in particles:
            pygame.draw.circle(screen, (255, 255, 255), particle.pos, particle.radius)
            pygame.draw.line(screen, (255, 255, 255), particle.pos,
                             (particle.pos[0] + particle.vel[0] * 10, particle.pos[1] + particle.vel[1] * 10), 1)

        pygame.display.flip()
