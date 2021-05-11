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
float3 operator/(float3 a, int b)
{
  float3 out;
  out.x = a.x/b;
  out.y = a.y/b;
  out.z = a.z/b;
  return out;
}

__device__
float3 operator*(float3 a, float b)
{
  float3 out;
  out.x = a.x*b;
  out.y = a.y*b;
  out.z = a.z*b;
  return out;
}

__device__
float3 operator+(float3 a, float3 b)
{
  float3 out;
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  out.z = a.z + b.z;
  return out;
}

__device__
float3 operator-(float3 a, float3 b)
{
  float3 out;
  out.x = a.x - b.x;
  out.y = a.y - b.y;
  out.z = a.z - b.z;
  return out;
}

__device__
uint3 operator+(uint3 a, uint3 b)
{
  uint3 out;
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  out.z = a.z + b.z;
  return out;
}

__device__
uint3 operator-(uint3 a, uint3 b)
{
  uint3 out;
  out.x = a.x - b.x;
  out.y = a.y - b.y;
  out.z = a.z - b.z;
  return out;
}

__device__
uint3 operator*(uint3 a, uint b)
{
  uint3 out;
  out.x = a.x * b;
  out.y = a.y * b;
  out.z = a.z * b;
  return out;
}

__device__
int dot(uint3 a, uint3 b)
{
  int out;
  out = a.x*b.x + a.y*b.y + a.z*b.z;
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

    uint3 sum;
    sum.x = 0;
    sum.y = 0;
    sum.z = 0;

    for (int offsetX = -sensorSize; offsetX <= sensorSize; offsetX++)
    {
      for (int offsetY = -sensorSize; offsetY <= sensorSize; offsetY++)
      {
        int2 pos;
        pos.x = sensorCentre.x + offsetX;
        pos.y = sensorCentre.y + offsetY;

        if(pos.x >= 0 && pos.x < WINDOW_WIDTH && pos.y >= 0 && pos.y < WINDOW_HEIGHT)
        {
          sum = sum + trailMap[pos.y * WINDOW_WIDTH + pos.x].val;
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

  uint senseVal;
  uint3 vect1;
  vect1.x = 1;
  vect1.y = 1;
  vect1.z = 1;

  if(sensorCentre.x >= 0 && sensorCentre.x < WINDOW_WIDTH && sensorCentre.y >= 0 && sensorCentre.y < WINDOW_HEIGHT)
  {
    senseVal = dot(trailMap[sensorCentre.y * WINDOW_WIDTH + sensorCentre.x].sense, agent->speciesMask * 2 - vect1);
  }

  return senseVal;
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
      agents[i].angle = randomSteerStrength * 2 * M_PI;
    }

    agents[i].position = newPos;
    trailMap[(uint) newPos.y * WINDOW_WIDTH + (uint) newPos.x].val = agents[i].speciesMask*255;
  }
}

__global__
void processTrailMap(uint n, struct TrailMap *trailMap, struct TrailMap *trailMapUpdated)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = index; i < n; i += stride)
  {
    float3 originalValue;
    originalValue.x = (float) trailMap[i].val.x;
    originalValue.y = (float) trailMap[i].val.y;
    originalValue.z = (float) trailMap[i].val.z;

    float3 sum;
    sum.x = 0;
    sum.y = 0;
    sum.z = 0;

    for (int offsetX = -1; offsetX <= 1; offsetX++)
    {
      for (int offsetY = -1; offsetY <= 1; offsetY++)
      {
        int sampleX = trailMap[i].x + offsetX;
        int sampleY = trailMap[i].y + offsetY;

        if(sampleX >= 0 && sampleX < WINDOW_WIDTH && sampleY >= 0 && sampleY < WINDOW_HEIGHT)
        {
          sum.x += trailMap[sampleY*WINDOW_WIDTH + sampleX].val.x;
          sum.y += trailMap[sampleY*WINDOW_WIDTH + sampleX].val.y;
          sum.z += trailMap[sampleY*WINDOW_WIDTH + sampleX].val.z;
        }

        float3 blurResult = sum / 9;

        float alpha = min(1.0f, DIFFUSE_SPEED * DELTA_TIME);
        float3 diffusedValue = originalValue*(1-alpha) + blurResult*alpha;
        float3 diffusedAndEvaporatedValue;
        diffusedAndEvaporatedValue.x = max(0.f, diffusedValue.x - EVAPORATE_SPEED * DELTA_TIME);
        diffusedAndEvaporatedValue.y = max(0.f, diffusedValue.y - EVAPORATE_SPEED * DELTA_TIME);
        diffusedAndEvaporatedValue.z = max(0.f, diffusedValue.z - EVAPORATE_SPEED * DELTA_TIME);

        trailMapUpdated[i].val.x = (uint) diffusedAndEvaporatedValue.x;
        trailMapUpdated[i].val.y = (uint) diffusedAndEvaporatedValue.y;
        trailMapUpdated[i].val.z = (uint) diffusedAndEvaporatedValue.z;
      }
    }
  }
}

__global__
void copyTrailMap(uint n, struct TrailMap *trailMap, struct TrailMap *trailMapUpdated)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = index; i < n; i += stride)
  {
    trailMap[i].val = trailMapUpdated[i].val;
  }
}

__global__
void setPixels(uint n, struct TrailMap *trailMap, sf::Uint8 *pixels)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = index; i < n; i += stride)
  {
    pixels[4*i] = trailMap[i].val.x;
    pixels[4*i+1] = trailMap[i].val.y;
    pixels[4*i+2] = trailMap[i].val.z;
    pixels[4*i+3] = 255;
  }
}

void CUDA::wrapper(uint n, struct Agent *agents, struct TrailMap *trailMap, struct TrailMap *trailMapUpdated, sf::Uint8 *pixels)
{
  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  senseMap<<<numBlocks, blockSize>>>(WINDOW_WIDTH*WINDOW_HEIGHT, trailMap);

  update<<<numBlocks, blockSize>>>(n, agents, trailMap);

  processTrailMap<<<numBlocks, blockSize>>>(WINDOW_WIDTH*WINDOW_HEIGHT, trailMap, trailMapUpdated);

  copyTrailMap<<<numBlocks, blockSize>>>(WINDOW_WIDTH*WINDOW_HEIGHT, trailMap, trailMapUpdated);

  setPixels<<<numBlocks, blockSize>>>(WINDOW_WIDTH*WINDOW_HEIGHT, trailMap, pixels);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}
