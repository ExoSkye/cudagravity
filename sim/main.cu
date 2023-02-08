#include <iostream>
#include "../common/definitions.hpp"
#include "../common/particle.hpp"
#include "cudaMemory.cuh"
#include "exportHelper.cuh"
#include "glm/glm.hpp"
#include <unistd.h>

__constant__ const float G_CONSTANT = 1;
__constant__ float dt = 0.0001;
__constant__ const float softening = 0.0000;
__constant__ const dim3 threadsPerBlock_gpu = dim3(1024);
const dim3 threadsPerBlock_cpu = dim3(1024);
__constant__ dim3 n_blocksPerGridGpu;

__global__ void getAccels(Particle* particles, vec2* accels, size_t N, size_t i) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        size_t j = id;

        if (j == i) {
            return;
        }

        Particle p2 = particles[i];
        Particle p1 = particles[j];

        float dist = glm::distance(p1.position, p2.position);

        vec2 f = -G_CONSTANT * ((p2.mass * p1.mass) / (dist * dist)) * glm::normalize(p2.position - p1.position);

        vec2 accel = f / p2.mass;

        atomicAdd(&(accels[i].x), accel.x);
        atomicAdd(&(accels[i].y), accel.y);
    }
}

__global__ void run_step(Particle* particles, vec2* accels, size_t N) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        Particle& cur_particle = particles[id];

        cur_particle.velocity.x += accels[id].x * dt / 2.0f;
        cur_particle.velocity.y += accels[id].y * dt / 2.0f;

        cur_particle.position.x += cur_particle.velocity.x * dt;
        cur_particle.position.y += cur_particle.velocity.y * dt;

        getAccels<<<n_blocksPerGridGpu, threadsPerBlock_gpu>>>(particles, accels, N, id);

        __syncthreads();

        cur_particle.velocity.x += accels[id].x * dt / 2.0f;
        cur_particle.velocity.y += accels[id].y * dt / 2.0f;
    }
}

float get_random() {
    float r = arc4random();

    return (r / ((float)UINT32_MAX / 2.0f)) - 1;
}

int main() {
    exportHelper exportHelper;

    size_t N = 2000;

    CudaMemory<Particle> particles = CudaMemory<Particle>(N);
    CudaMemory<vec2> accelerations = CudaMemory<vec2>(N);

    for (int i = 0; i < N; i++) {
        accelerations[i] = vec2(0, 0);
        particles[i] = Particle{10, vec2(get_random(), get_random()), vec2(get_random(), get_random())};
    }

    dim3 n_blocksPerGrid = dim3(
            ((N) + threadsPerBlock_cpu.x - 1) / threadsPerBlock_cpu.x
    );

    printf("n_blocksPerGrid: %i\n", n_blocksPerGrid.x);
    printf("threadsPerBlock: %i\n", threadsPerBlock_cpu.x);

    cudaMemcpyToSymbol(n_blocksPerGridGpu, &n_blocksPerGrid, sizeof(dim3));
    cudaMemcpyToSymbol(threadsPerBlock_gpu, &threadsPerBlock_cpu, sizeof(dim3));

    particles.send();
    accelerations.send();

    exportHelper.setParticles(&particles, N);

    while(true) {
        run_step<<<n_blocksPerGrid, threadsPerBlock_cpu>>>(particles.getDevicePointer(), accelerations.getDevicePointer(), N);

        cudaDeviceSynchronize();

        accelerations.send();

        exportHelper.epoch();

        if (!exportHelper.should_stop) {
            break;
        }
    }

    exportHelper.stop();

    cudaDeviceSynchronize();

    return 0;
}
