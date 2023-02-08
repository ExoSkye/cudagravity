#include <cstdlib>
#include <SDL2/SDL.h>
#include <SDL2_rotozoom.h>
#include <SDL2_framerate.h>
#include <vector>
#include <unistd.h>
#include "definitions.hpp"
#include "particle.hpp"


int main() {
    double dt = 0.01;
    double t = 1.0;
    size_t current_idx = 1;
    bool pause = true;

    // Toggles

    bool show_fps = false;
    bool show_help = true;
    bool show_zoom = true;
    bool show_state = true;
    bool show_speed = true;
    bool show_grid = true;
    bool auto_size = true;
    bool shift = false;
    bool ctrl = false;

    bool update_required = true;

    // Init SDL

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("Verlet", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 800, 800,
                                          SDL_WINDOW_SHOWN);

    SDL_Event event;

    SDL_Surface *surf = SDL_CreateRGBSurface(0, 800, 800, 32, 0, 0, 0, 0);

    std::vector<Particle> particles;

    FPSmanager fps;

    SDL_initFramerate(&fps);
    SDL_setFramerate(&fps, 60);

    float max_dim = 0.0;

    while (true) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                exit(0);
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_RSHIFT:
                    case SDLK_LSHIFT:
                        shift = true;
                        break;

                    case SDLK_RCTRL:
                    case SDLK_LCTRL:
                        ctrl = true;
                        break;

                    case SDLK_SPACE:
                        pause = !pause;
                        printf("Paused: %s\n", pause ? "true" : "false");
                        break;

                    case SDLK_LEFT:
                        t -= dt;
                        current_idx = (int) t;
                        update_required = true;
                        break;

                    case SDLK_RIGHT:
                        t += dt;
                        current_idx = (int) t;
                        update_required = true;
                        break;

                    case SDLK_KP_PLUS:
                        dt *= 2;
                        printf("dt: %f\n", dt);
                        break;

                    case SDLK_KP_MINUS:
                        dt /= 2;
                        printf("dt: %f\n", dt);
                        break;

                    case SDLK_KP_0:
                        t = 1;
                        current_idx = 1;
                        update_required = true;
                        break;

                    case SDLK_UP:
                        if (shift) {
                            max_dim -= 1;
                        } else if (ctrl) {
                            max_dim -= 10;
                        } else {
                            max_dim -= 0.1;
                        }

                        max_dim = std::abs(max_dim);

                        update_required = true;
                        break;

                    case SDLK_DOWN:
                        if (shift) {
                            max_dim += 1;
                        } else if (ctrl) {
                            max_dim += 10;
                        } else {
                            max_dim += 0.1;
                        }
                        update_required = true;
                        break;

                    case SDLK_a:
                        auto_size = !auto_size;
                        update_required = true;
                        printf("Auto size: %s\n", auto_size ? "true" : "false");
                        break;
                }
            } else if (event.type == SDL_KEYUP) {
                switch (event.key.keysym.sym) {
                    case SDLK_RSHIFT:
                    case SDLK_LSHIFT:
                        shift = false;
                        break;

                    case SDLK_RCTRL:
                    case SDLK_LCTRL:
                        ctrl = false;
                        break;
                }
            }
        }

        if (!pause) {
            t += dt;
            current_idx = (int) t;
            update_required = true;
        }

        if (update_required) {
            char filename[256];
            sprintf(filename, "./output/%zu.dat", current_idx);

            FILE *f = fopen(filename, "rb");

            if (f == nullptr) {
                printf("File not found: %s\n", filename);
                printf("End of simulation\n");
                pause = true;

            } else {
                SDL_FillRect(surf, nullptr, SDL_MapRGB(surf->format, 0, 0, 0));

                particles.clear();

                char *cur_line = (char *) malloc(sizeof(float) * 5 + 1);
                size_t len = sizeof(float) * 5 + 1;

                while (true) {
                    fread(cur_line, len, 1, f);

                    if (feof(f)) {
                        break;
                    }

                    float x = *(float *) cur_line;
                    float y = *(float *)&cur_line[sizeof(float)];
                    float vx = *(float *)&cur_line[sizeof(float) * 2];
                    float vy = *(float *)&cur_line[sizeof(float) * 3];
                    float m = *(float *)&cur_line[sizeof(float) * 4];

                    particles.push_back(Particle{m, {x, y}, {vx, vy}});
                }

                fclose(f);

                if (auto_size) {
                    for (auto &particle: particles) {
                        max_dim = std::max(max_dim,
                                           std::max(std::abs(particle.position.x), std::abs(particle.position.y)));
                    }
                }

                for (auto &particle: particles) {
                    particle.position /= (max_dim * 1.1);
                    particle.position *= 400.0;
                    particle.position += vec2(400, 400);

                    SDL_Rect rect = {(int) particle.position.x - 1, (int) particle.position.y - 1, 2, 2};

                    SDL_FillRect(surf, &rect, SDL_MapRGB(surf->format, 255, 255, 255));
                }
            }

            update_required = false;
        }

        SDL_framerateDelay(&fps);

        SDL_Surface *window_surface = SDL_GetWindowSurface(window);

        SDL_BlitSurface(surf, NULL, window_surface, NULL);

        SDL_UpdateWindowSurface(window);
    }

    return 0;
}
