cmake_minimum_required(VERSION 3.10)
project(verlet CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(SDL2 REQUIRED)

add_subdirectory(glm)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-G")

add_executable(cudagravity main.cu particle.cu particle.cuh definitions.cuh cudaMemory.cuh sdlHelper.cu sdlHelper.cuh)

target_link_libraries(cudagravity PUBLIC glm::glm SDL2::SDL2)

set_target_properties(cudagravity PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
