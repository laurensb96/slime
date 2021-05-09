#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <SFML/Graphics.hpp>
#include <math.h>

#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

#define NUM_AGENTS 1e6
#define MOVE_SPEED 25.f
#define DELTA_TIME 0.02f
#define EVAPORATE_SPEED 100.f
#define TURN_SPEED 10.f

struct Agent {
    float2 position;
    float angle;
};

struct TrailMap {
    uint x, y;
    sf::Uint8 val;
    uint sense;
};

namespace CUDA {
    void wrapper(uint n, struct Agent *agents, struct TrailMap *trailMap, sf::Uint8 *pixels);
}

