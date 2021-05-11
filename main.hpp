#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <SFML/Graphics.hpp>
#include <math.h>

#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

#define NUM_AGENTS 900e3;
#define DELTA_TIME 0.04f

#define MOVE_SPEED 50.f
#define TURN_SPEED 50.f
#define DIFFUSE_SPEED 20.f
#define EVAPORATE_SPEED 20.f

struct Agent {
    float2 position;
    float angle;
    uint3 speciesMask;
};

struct TrailMap {
    uint x, y;
    uint3 val;
    uint3 sense;
};

namespace CUDA {
    void wrapper(uint n, struct Agent *agents, struct TrailMap *trailMap, struct TrailMap *trailMapUpdated, sf::Uint8 *pixels);
}

