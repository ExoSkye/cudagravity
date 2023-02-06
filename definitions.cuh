#include <glm/vec2.hpp>

typedef glm::vec<2, float, glm::defaultp> vec2;
#define CUDA_CALL(CALL) if (CALL != cudaSuccess) { printf("CUDA call failed at %s:%i\n", __FILE__,__LINE__); exit(-1); }
