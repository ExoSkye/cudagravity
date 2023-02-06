

#ifndef VERLET_PARTICLE_CUH
#define VERLET_PARTICLE_CUH

#include "definitions.cuh"

class Particle {
public:
    float mass;
    vec2 position;
    vec2 velocity;
};


#endif //VERLET_PARTICLE_CUH
