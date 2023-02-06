#include <iostream>
#include "definitions.cuh"
#include "particle.cuh"
#include "cudaMemory.cuh"
#include "sdlHelper.cuh"
#include <glm/glm.hpp>
#include <random>

__constant__ const float G_CONSTANT = 1;
__constant__ float dt = 0.1;
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

        Particle particle = particles[i];
        Particle other_particle = particles[j];

        vec2 d = {
                other_particle.position.x - particle.position.x,
                other_particle.position.y - particle.position.y
        };

        float dist = glm::distance(particle.position, other_particle.position);

        vec2 f = -G_CONSTANT * ((particle.mass * other_particle.mass) / (dist * dist)) * glm::normalize(other_particle.position - particle.position);

        vec2 accel = f / particle.mass;

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

int main() {
    sdlHelper sdlHelper;

    size_t N = 20000;

    CudaMemory<Particle> particles = CudaMemory<Particle>(N);
    CudaMemory<vec2> accelerations = CudaMemory<vec2>(N);

    std::default_random_engine generator(std::random_device{}());
    std::uniform_real_distribution<float> mass_distribution(1.0, 100000.0);
    std::uniform_real_distribution<float> pos_distribution(-0.1, 0.1);
    std::uniform_real_distribution<float> vel_distribution(-2.0, 2.0);

    for (size_t i = 0; i < N; i++) {
        accelerations[i] = {0, 0};
        particles[i] = {
                mass_distribution(generator),
                {pos_distribution(generator), pos_distribution(generator)},
                {vel_distribution(generator), vel_distribution(generator)}
        };
    }

    cudaDeviceGetAttribute((int*)(&threadsPerBlock_cpu.x), cudaDevAttrMaxThreadsPerBlock, 0);

    dim3 n_blocksPerGrid = dim3(
            ((N) + threadsPerBlock_cpu.x - 1) / threadsPerBlock_cpu.x
    );

    printf("n_blocksPerGrid: %i\n", n_blocksPerGrid.x);
    printf("threadsPerBlock: %i\n", threadsPerBlock_cpu.x);

    cudaMemcpyToSymbol(n_blocksPerGridGpu, &n_blocksPerGrid, sizeof(dim3));
    cudaMemcpyToSymbol(threadsPerBlock_gpu, &threadsPerBlock_cpu, sizeof(dim3));

    particles.send();
    accelerations.send();

    for (int i = 0; i < 2; i++) {
        run_step<<<n_blocksPerGrid, threadsPerBlock_cpu>>>(particles.getDevicePointer(), accelerations.getDevicePointer(), N);

        cudaDeviceSynchronize();
        particles.sync();
        accelerations.send();

        if (!sdlHelper.drawParticles(particles.getPointer(), N)) {
            break;
        }
    }

    cudaDeviceSynchronize();

    return 0;
}
