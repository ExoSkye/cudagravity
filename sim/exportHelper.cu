//
// Created by kai on 05/02/23.
//

#include "exportHelper.cuh"

#include <sys/stat.h>

exportHelper::exportHelper() {
    mkdir("output", S_IRWXU);

    char* epochs_str = getenv("EPOCHS");

    if (epochs_str != nullptr) {
        if (sscanf(epochs_str, "%zu", &this->n) == 0) {
            printf("Failed to parse EPOCHS environment variable - required when NO_DISPLAY is set\n");
            should_stop = false;
        }
    } else {
        printf("Failed to parse EPOCHS environment variable - required when NO_DISPLAY is set\n");
        should_stop = false;
    }

    if (instance == nullptr) {
        instance = this;
    }
}

void exportHelper::stop() {
    if (instance == this) {
        should_stop = false;

        for (auto& thread : exportThreads) {
            if (thread.running) {
                thread.thread.join();
            }
        }

    }
}

exportHelper::~exportHelper() {
    stop();
}

exportHelper *exportHelper::getInstance() {
    return this->instance;
}

void exportHelper::imageWriter(size_t idx, thread* thread) {
    char filename[256];
    sprintf(filename, "output/%zu.dat", idx);

    FILE* f = fopen64(filename, "wb");

    auto* positions_copy = (vec2*)malloc(this->particle_count * sizeof(vec2));
    auto* velocities_copy = (vec2*)malloc(this->particle_count * sizeof(vec2));
    auto* masses_copy = (float*)malloc(this->particle_count * sizeof(float));

    this->positions->sync_to(&positions_copy);
    this->velocities->sync_to(&velocities_copy);
    this->masses->sync_to(&masses_copy);

    char data[sizeof(float) * 5 + 1];
    data[sizeof(float) * 5] = '\n';

    for (size_t i = 0; i < particle_count; i++) {
	    memcpy(data, &positions_copy[i].x, sizeof(float));
        memcpy(&data[sizeof(float)], &positions_copy[i].y, sizeof(float));
        memcpy(&data[sizeof(float) * 2], &velocities_copy[i].x, sizeof(float));
        memcpy(&data[sizeof(float) * 3], &velocities_copy[i].y, sizeof(float));
        memcpy(&data[sizeof(float) * 4], &masses_copy[i], sizeof(float));

        fwrite(data, sizeof(char), sizeof(float) * 5 + 1, f);
    }

    free(positions_copy);
    free(velocities_copy);
    free(masses_copy);
    fflush(f);
    fclose(f);

    thread->running = false;
}

void exportHelper::epoch() {
    this->count++;

    if (this->count % 100 == 0) {
        printf("Epoch: %zu\n", this->count);
    }

    if (this->n != 0 && this->count >= this->n) {
        this->should_stop = false;
    }

    this->exportThreads.emplace_back();

    this->exportThreads.back().thread = std::thread(&exportHelper::imageWriter, this, this->count, &this->exportThreads.back());

    for (auto it = this->exportThreads.begin(); it != this->exportThreads.end();) {
        if (!it->running) {
            it->thread.join();
            it = this->exportThreads.erase(it);
        } else {
            it++;
        }
    }

}
