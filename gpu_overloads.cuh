#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
