#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <SFML/Graphics.hpp>
#include <math.h>

#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

#define NUM_AGENTS 1e6
#define DELTA_TIME 0.04f

#define MOVE_SPEED 50.f
#define TURN_SPEED 25.f
#define DIFFUSE_SPEED 10.f
#define EVAPORATE_SPEED 20.f

struct Agent
{
    float4 position;
    float angle;
    int4 speciesMask;
};

struct TrailMap
{
    int x, y;
    int4 val;
    int4 sense;
};

namespace CUDA
{
void wrapper(struct Agent *agents, struct TrailMap *trailMap, struct TrailMap *trailMapUpdated, sf::Uint8 *pixels);
}

