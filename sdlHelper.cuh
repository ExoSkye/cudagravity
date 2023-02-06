//
// Created by kai on 05/02/23.
//

#ifndef VERLET_SDLHELPER_CUH
#define VERLET_SDLHELPER_CUH

#include <SDL2/SDL.h>
#include <vector>
#include <thread>
#include "particle.cuh"

class sdlHelper {
public:
    sdlHelper();
    ~sdlHelper();

    bool drawParticles(Particle* particles, size_t particleCount);
    sdlHelper* getInstance();
private:
    void renderThreadFunc();

    sdlHelper* instance = nullptr;
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    bool running = true;
    bool displaying = true;

    std::thread renderThread;

    std::vector<SDL_Rect> particleRects;
};


#endif //VERLET_SDLHELPER_CUH
