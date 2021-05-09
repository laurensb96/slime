#include "slime.cuh"

__device__
float2 operator+(float2 a, float2 b)
{
  float2 out;
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  return out;
}

__device__
float2 operator*(float2 a, float b)
{
  float2 out;
  out.x = a.x*b;
  out.y = a.y*b;
  return out;
}

__device__
uint hash(uint state)
{
  state ^= 2747636419u;
  state *= 2654435769u;
  state ^= state >> 16;
  state *= 2654435769u;
  state ^= state >> 16;
  state *= 2654435769u;
  return state;
}

__global__
void senseMap(uint n, struct TrailMap *trailMap)
{
  static int sensorSize = 2;

  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = index; i < n; i += stride)
  {
    int2 sensorCentre;
    sensorCentre.x = trailMap[i].x;
    sensorCentre.y = trailMap[i].y;

    uint sum = 0;

    for (int offsetX = -sensorSize; offsetX <= sensorSize; offsetX++)
    {
      for (int offsetY = -sensorSize; offsetY <= sensorSize; offsetY++)
      {
        int2 pos;
        pos.x = sensorCentre.x + offsetX;
        pos.y = sensorCentre.y + offsetY;

        if(pos.x >= 0 && pos.x < WINDOW_WIDTH && pos.y >= 0 && pos.y < WINDOW_HEIGHT)
        {
          sum += trailMap[pos.y * WINDOW_WIDTH + pos.x].val;
        }
      }
    }

    trailMap[i].sense = sum;
  }
}

__device__
uint sense(struct Agent *agent, float sensorAngleOffset, struct TrailMap *trailMap)
{
  static int sensorOffsetDst = 20;

  float sensorAngle = agent->angle + sensorAngleOffset;
  float2 sensorDir;
  sensorDir.x = cosf(sensorAngle);
  sensorDir.y = sinf(sensorAngle);
  int2 sensorCentre;
  sensorCentre.x = agent->position.x + sensorDir.x * sensorOffsetDst;
  sensorCentre.y = agent->position.y + sensorDir.y * sensorOffsetDst;
  
  uint sum;

  if(sensorCentre.x >= 0 && sensorCentre.x < WINDOW_WIDTH && sensorCentre.y >= 0 && sensorCentre.y < WINDOW_HEIGHT)
  {
    sum = trailMap[sensorCentre.y * WINDOW_WIDTH + sensorCentre.x].sense;
  }

  return sum;
}

__global__
void update(uint n, struct Agent *agents, struct TrailMap *trailMap)
{
  static float sensorAngleSpacing = (float) (30 * M_PI/180);

  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = index; i < n; i += stride)
  {
    Agent agent = agents[i];

    uint random = hash(agent.position.y * WINDOW_WIDTH + agent.position.x + hash(i % WINDOW_WIDTH));
  
    uint weightForward = sense(&agent, 0, trailMap);
    uint weightLeft = sense(&agent, sensorAngleSpacing, trailMap);
    uint weightRight = sense(&agent, -sensorAngleSpacing, trailMap);

    float randomSteerStrength = random/4294967295.0;

    if (weightForward > weightLeft && weightForward > weightRight)
    {
      agents[i].angle += 0;
    }
    else if (weightForward < weightLeft && weightForward < weightRight)
    {
      agents[i].angle += (randomSteerStrength - 0.5) * 2 * TURN_SPEED * DELTA_TIME;
    }
    else if (weightRight > weightLeft)
    {
      agents[i].angle -= randomSteerStrength * TURN_SPEED * DELTA_TIME;
    }
    else if (weightLeft > weightRight)
    {
      agents[i].angle += randomSteerStrength * TURN_SPEED * DELTA_TIME;
    }

    float2 direction, newPos;
    direction.x = cosf(agent.angle);
    direction.y = sinf(agent.angle);

    newPos = agent.position + direction * MOVE_SPEED * DELTA_TIME;

    if(newPos.x < 0 || newPos.x >= WINDOW_WIDTH || newPos.y < 0 || newPos.y >= WINDOW_HEIGHT)
    {
      newPos.x = min(WINDOW_WIDTH-0.01f, max(0.f, newPos.x));
      newPos.y = min(WINDOW_HEIGHT-0.01f, max(0.f, newPos.y));
      agents[i].angle = (random/4294967295.0) * 2 * M_PI;
    }

    agents[i].position = newPos;
    trailMap[(uint) newPos.y * WINDOW_WIDTH + (uint) newPos.x].val = 255;
  }
}

__global__
void processTrailMap(uint n, struct TrailMap *trailMap)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = index; i < n; i += stride)
  {
    float originalValue = trailMap[i].val;

    float sum = 0;
    for (int offsetX = -1; offsetX <= 1; offsetX++)
    {
      for (int offsetY = -1; offsetY <= 1; offsetY++)
      {
        int sampleX = trailMap[i].x + offsetX;
        int sampleY = trailMap[i].y + offsetY;

        if(sampleX >= 0 && sampleX < WINDOW_WIDTH && sampleY >= 0 && sampleY < WINDOW_HEIGHT)
        {
          sum += trailMap[i].val;
        }

        float blurResult = sum / 9;

        float alpha = min(1.0f, DIFFUSE_SPEED * DELTA_TIME);
        float diffusedValue = originalValue*(1-alpha) + blurResult*alpha;
        float diffusedAndEvaporatedValue = max(0.f, diffusedValue - EVAPORATE_SPEED * DELTA_TIME);

        trailMap[i].val = diffusedAndEvaporatedValue;
      }
    }
  }
}

__global__
void setPixels(uint n, struct TrailMap *trailMap, sf::Uint8 *pixels)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = index; i < n; i += stride)
  {
    pixels[4*i] = pow(trailMap[i].val,3)/(255*255)*0.8;
    pixels[4*i+1] = pow(trailMap[i].val,2)/255*0.8;
    pixels[4*i+2] = trailMap[i].val*0.6;
    pixels[4*i+3] = 255;
  }
}

void CUDA::wrapper(uint n, struct Agent *agents, struct TrailMap *trailMap, sf::Uint8 *pixels)
{
  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  senseMap<<<numBlocks, blockSize>>>(WINDOW_WIDTH*WINDOW_HEIGHT, trailMap);

  update<<<numBlocks, blockSize>>>(n, agents, trailMap);

  processTrailMap<<<numBlocks, blockSize>>>(WINDOW_WIDTH*WINDOW_HEIGHT, trailMap);

  setPixels<<<numBlocks, blockSize>>>(WINDOW_WIDTH*WINDOW_HEIGHT, trailMap, pixels);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}