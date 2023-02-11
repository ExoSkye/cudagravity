//
// Created by kai on 05/02/23.
//

#ifndef VERLET_EXPORTHELPER_CUH
#define VERLET_EXPORTHELPER_CUH

#include <vector>
#include <thread>
#include <list>
#include <atomic>
#include <particle.hpp>
#include "cudaMemory.cuh"

struct thread {
    std::atomic<bool> running = true;
    std::thread thread = std::thread();
};

class exportHelper {
public:
    bool should_stop = true;

    explicit exportHelper(size_t epochs);
    ~exportHelper();

    exportHelper* getInstance();

    void setParticles(CudaMemory<vec2>* _velocities, CudaMemory<vec2>* _positions, CudaMemory<float>* _masses, size_t N) {
        this->velocities = _velocities;
        this->positions = _positions;
        this->masses = _masses;
        this->particle_count = N;
    }

    void epoch();

    void stop();
private:
    exportHelper* instance = nullptr;
    bool displaying = true;

    size_t n = 0;
    size_t count = 0;

    CudaMemory<vec2>* velocities;
    CudaMemory<vec2>* positions;
    CudaMemory<float>* masses;

    size_t particle_count;

    std::list<thread> exportThreads;

    void imageWriter(size_t idx, thread* thread);
};


#endif //VERLET_EXPORTHELPER_CUH
