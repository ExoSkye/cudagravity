#include <iostream>
#include "../common/definitions.hpp"
#include "../common/particle.hpp"
#include "cudaMemory.cuh"
#include "exportHelper.cuh"
#include "glm/glm.hpp"
#include <unistd.h>

__constant__ float G_CONSTANT = 1;
__constant__ float dt = 0.04366762767265625454879349793878376839702869868857256785727591752676;
__constant__ dim3 threadsPerBlock_gpu = dim3(1024);
dim3 threadsPerBlock_cpu = dim3(1024);
__constant__ dim3 n_blocksPerGridGpu;

__global__ void getAccels(vec2* positions, const float* masses, vec2* accels, size_t N, size_t i) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        size_t j = id;

        if (j == i) {
            return;
        }

        vec2 p1 = positions[j];
        vec2 p2 = positions[i];

        float m1 = masses[j];
        float m2 = masses[i];

        float dist = glm::distance(p1, p2);

        vec2 f = -G_CONSTANT * ((m2 * m1) / (dist * dist)) * glm::normalize(p2 - p1);

        vec2 accel = f / m2;

        atomicAdd(&(accels[i].x), accel.x);
        atomicAdd(&(accels[i].y), accel.y);
    }
}

__global__ void run_step(vec2* positions, vec2* velocities, const float* masses, vec2* accels, size_t N) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        vec2& position = positions[id];
        vec2& velocity = velocities[id];

        velocity.x += accels[id].x * dt / 2.0f;
        velocity.y += accels[id].y * dt / 2.0f;

        position.x += velocity.x * dt;
        position.y += velocity.y * dt;

        getAccels<<<n_blocksPerGridGpu, threadsPerBlock_gpu>>>(positions, masses, accels, N, id);

        __syncthreads();

        velocity.x += accels[id].x * dt / 2.0f;
        velocity.y += accels[id].y * dt / 2.0f;
    }
}

int main() {
    FILE *fp = fopen("initial.ibin", "rb");

    if (fp == NULL) {
        printf("Initial file not found\n");
        return 1;
    }

    char *cur_line = (char *) malloc(sizeof(float) * 5);
    size_t len = sizeof(float) * 5;

    size_t N = 0;

    // Read Header

    char magic[4];
    fread(magic, sizeof(char), 4, fp);

    if (strcmp(magic, "IBIN") != 0) {
        printf("Invalid file format\n");
        return 1;
    }

    size_t epochs;

    fread(&dt, sizeof(float), 1, fp);
    fread(&(threadsPerBlock_cpu.x), sizeof(int), 1, fp);
    fread(&epochs, sizeof(size_t), 1, fp);
    fread(&G_CONSTANT, sizeof(float), 1, fp);
    fread(&N, sizeof(size_t), 1, fp);
    fseek(fp, 4, SEEK_CUR);

    exportHelper exportHelper(epochs);

    CudaMemory<float> masses = CudaMemory<float>(N);
    CudaMemory<vec2> velocities = CudaMemory<vec2>(N);
    CudaMemory<vec2> positions = CudaMemory<vec2>(N);
    CudaMemory<vec2> accelerations = CudaMemory<vec2>(N);

    for (size_t i = 0; i < N; i++) {
        fread(cur_line, len, 1, fp);

        float x = *(float *) cur_line;
        float y = *(float *)&cur_line[sizeof(float)];
        float vx = *(float *)&cur_line[sizeof(float) * 2];
        float vy = *(float *)&cur_line[sizeof(float) * 3];
        float m = *(float *)&cur_line[sizeof(float) * 4];

        masses[i] = m;
        velocities[i] = vec2(vx, vy);
        positions[i] = vec2(x, y);
    }

    dim3 n_blocksPerGrid = dim3(
            ((N) + threadsPerBlock_cpu.x - 1) / threadsPerBlock_cpu.x
    );

    printf("n_blocksPerGrid: %i\n", n_blocksPerGrid.x);
    printf("threadsPerBlock: %i\n", threadsPerBlock_cpu.x);

    cudaMemcpyToSymbol(n_blocksPerGridGpu, &n_blocksPerGrid, sizeof(dim3));
    cudaMemcpyToSymbol(threadsPerBlock_gpu, &threadsPerBlock_cpu, sizeof(dim3));

    velocities.send();
    positions.send();
    masses.send();
    accelerations.send();

    exportHelper.setParticles(&velocities, &positions, &masses, N);

    while(true) {
        run_step<<<n_blocksPerGrid, threadsPerBlock_cpu>>>(positions.getDevicePointer(), velocities.getDevicePointer(), masses.getDevicePointer(), accelerations.getDevicePointer(), N);

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
