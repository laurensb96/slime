#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__
float4 operator+(float4 a, float4 b)
{
    float4 out;
    out.x = a.x + b.x;
    out.y = a.y + b.y;
    out.z = a.z + b.z;
    out.w = a.z + b.w;
    return out;
}

__device__
float4 operator-(float4 a, float4 b)
{
    float4 out;
    out.x = a.x - b.x;
    out.y = a.y - b.y;
    out.z = a.z - b.z;
    out.w = a.w - b.w;
    return out;
}

__device__
float4 operator*(float4 a, float b)
{
    float4 out;
    out.x = a.x*b;
    out.y = a.y*b;
    out.z = a.z*b;
    out.w = a.w*b;
    return out;
}

__device__
float4 operator/(float4 a, float b)
{
    float4 out;
    out.x = a.x/b;
    out.y = a.y/b;
    out.z = a.z/b;
    out.w = a.w/b;
    return out;
}

__device__
int4 operator+(int4 a, int4 b)
{
    int4 out;
    out.x = a.x + b.x;
    out.y = a.y + b.y;
    out.z = a.z + b.z;
    out.w = a.z + b.w;
    return out;
}

__device__
int4 operator-(int4 a, int4 b)
{
    int4 out;
    out.x = a.x - b.x;
    out.y = a.y - b.y;
    out.z = a.z - b.z;
    out.w = a.w - b.w;
    return out;
}

__device__
int4 operator*(int4 a, int b)
{
    int4 out;
    out.x = a.x*b;
    out.y = a.y*b;
    out.z = a.z*b;
    out.w = a.w*b;
    return out;
}

__device__
float4 setFloat4(float x, float y, float z, float w)
{
    float4 out;

    out.x = x;
    out.y = y;
    out.z = z;
    out.w = w;

    return out;
}

__device__
int4 setInt4(int x, int y, int z, int w)
{
    int4 out;

    out.x = x;
    out.y = y;
    out.z = z;
    out.w = w;

    return out;
}
