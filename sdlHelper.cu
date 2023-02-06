//
// Created by kai on 05/02/23.
//

#include "sdlHelper.cuh"

sdlHelper::sdlHelper() {
    if (getenv("NO_DISPLAY") != nullptr) {
        this->displaying = false;
        return;
    }

    if (instance == nullptr) {
        instance = this;

        this->renderThread = std::thread(&sdlHelper::renderThreadFunc, this);
    }
}

sdlHelper::~sdlHelper() {
    if (instance == this) {
        running = false;

        this->renderThread.join();
        SDL_Quit();
    }
}

bool sdlHelper::drawParticles(Particle* particles, size_t particleCount) {
    if (!this->displaying) return this->running;
    this->particleRects.clear();
    this->particleRects.reserve(particleCount);

    float maxDistance = 0;

    for (int i = 0; i < particleCount; i++) {
        maxDistance = std::max(maxDistance, particles[i].position.x);
        maxDistance = std::max(maxDistance, particles[i].position.y);
    }

    for (int i = 0; i < particleCount; i++) {
        SDL_Rect rect;

        vec2 pos = particles[i].position;
        pos /= (maxDistance * 1.1);
        pos *= 512;
        pos += vec2(512, 512);

        rect.x = pos.x - 1;
        rect.y = pos.y - 1;
        rect.w = 2;
        rect.h = 2;
        this->particleRects.push_back(rect);
    }
    return this->running;
}

sdlHelper *sdlHelper::getInstance() {
    return this->instance;
}

void sdlHelper::renderThreadFunc() {
    this->window = SDL_CreateWindow("cudaGravity", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1024, 1024, SDL_WINDOW_VULKAN);
    this->renderer = SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    while (running) {
        SDL_Event e;

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running = false;
            }
        }

        SDL_SetRenderDrawColor(this->renderer, 0, 0, 0, 255);
        SDL_RenderClear(this->renderer);
        SDL_SetRenderDrawColor(this->renderer, 255, 255, 255, 255);

        for (auto & particleRect : particleRects) {
            SDL_RenderFillRect(this->renderer, &particleRect);
        }

        SDL_RenderPresent(this->renderer);
    }

    SDL_DestroyRenderer(this->renderer);
    SDL_DestroyWindow(this->window);
}
