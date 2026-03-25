#pragma once
#include "math.cuh"

namespace Rasterizer
{
    struct Vertex
    {
        Math::Vec3 Position;
        Math::Vec3 Color;    // RGB [0,1]
    };

    // Vertex shader: transform each vertex by the MVP matrix, do perspective divide,
    // then map from NDC [-1,1] to screen coordinates [0, width/height].
    __global__
    void VertexShader(
        const Vertex* vertices,
        Math::Vec3* screenPositions,   // output: screen-space XY + NDC depth in Z
        int vertexCount,
        Math::Mat4 mvp,
        int width,
        int height)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= vertexCount) return;

        // Transform to clip space
        Math::Vec4 clip = mvp * Math::Vec4(vertices[i].Position, 1.0f);

        // Perspective divide → NDC (X,Y in [-1,1], Z in [0,1])
        Math::Vec3 ndc = clip.PerspectiveDivide();

        // Viewport transform: NDC → screen pixels
        // X: [-1,1] → [0, width], Y: [-1,1] → [height, 0] (flip Y so +Y is up)
        float sx = (ndc.X * 0.5f + 0.5f) * static_cast<float>(width);
        float sy = (1.0f - (ndc.Y * 0.5f + 0.5f)) * static_cast<float>(height);

        screenPositions[i] = { sx, sy, ndc.Z };
    }

    // Edge function: positive when point P is on the left side of edge V0→V1
    __device__
    auto EdgeFunction(Math::Vec3 v0, Math::Vec3 v1, float px, float py) -> float
    {
        return (v1.X - v0.X) * (py - v0.Y) - (v1.Y - v0.Y) * (px - v0.X);
    }

    // Rasterizer + fragment shader: one thread per pixel.
    // For each pixel, test against all triangles. This is the brute-force approach —
    // simple and correct, good for learning.
    __global__
    void Rasterize(
        const Vertex* vertices,
        const Math::Vec3* screenPositions,
        int triangleCount,                // number of triangles (vertexCount / 3)
        unsigned char* framebuffer,        // RGBA, width * height * 4
        int* depthBuffer,                  // integer depth buffer (higher = farther)
        int width,
        int height)
    {
        int px = blockIdx.x * blockDim.x + threadIdx.x;
        int py = blockIdx.y * blockDim.y + threadIdx.y;
        if (px >= width || py >= height) return;

        float pixelX = static_cast<float>(px) + 0.5f;
        float pixelY = static_cast<float>(py) + 0.5f;

        for (int tri = 0; tri < triangleCount; tri++)
        {
            int i0 = tri * 3;
            int i1 = tri * 3 + 1;
            int i2 = tri * 3 + 2;

            Math::Vec3 s0 = screenPositions[i0];
            Math::Vec3 s1 = screenPositions[i1];
            Math::Vec3 s2 = screenPositions[i2];

            // Bounding box clamp
            float minX = fminf(fminf(s0.X, s1.X), s2.X);
            float maxX = fmaxf(fmaxf(s0.X, s1.X), s2.X);
            float minY = fminf(fminf(s0.Y, s1.Y), s2.Y);
            float maxY = fmaxf(fmaxf(s0.Y, s1.Y), s2.Y);

            // Early out: pixel not in triangle's bounding box
            if (pixelX < minX || pixelX > maxX || pixelY < minY || pixelY > maxY)
                continue;

            // Edge function test
            float w0 = EdgeFunction(s1, s2, pixelX, pixelY);
            float w1 = EdgeFunction(s2, s0, pixelX, pixelY);
            float w2 = EdgeFunction(s0, s1, pixelX, pixelY);

            // Check if pixel is inside triangle (all same sign).
            // After the viewport Y-flip, CCW triangles become CW in screen space,
            // so edge functions may all be negative. Accept both windings.
            bool inside = (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                       || (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
            if (inside)
            {
                // Barycentric coordinates (normalize).
                // For CW winding, area and weights are both negative, so the
                // division produces correct positive barycentric coordinates.
                float area = w0 + w1 + w2;
                if (fabsf(area) < 1e-6f) continue;
                float invArea = 1.0f / area;
                w0 *= invArea;
                w1 *= invArea;
                w2 *= invArea;

                // Interpolate depth
                float depth = w0 * s0.Z + w1 * s1.Z + w2 * s2.Z;

                // Map depth to integer for atomic comparison (higher int = farther away).
                // We want the closest fragment (smallest depth) to win.
                // Since atomicMin picks the smallest, and smaller depth = closer, this works directly.
                int intDepth = __float_as_int(depth);

                // Handle negative depths: make all values comparable via atomicMin.
                // IEEE 754 floats: positive floats compare correctly as ints,
                // but negative floats are inverted. Since depth should be in [0,1], this is fine.

                int pixelIndex = py * width + px;
                int oldDepth = atomicMin(&depthBuffer[pixelIndex], intDepth);

                if (intDepth <= oldDepth)
                {
                    // Interpolate color
                    Math::Vec3 c0 = vertices[i0].Color;
                    Math::Vec3 c1 = vertices[i1].Color;
                    Math::Vec3 c2 = vertices[i2].Color;

                    float r = w0 * c0.X + w1 * c1.X + w2 * c2.X;
                    float g = w0 * c0.Y + w1 * c1.Y + w2 * c2.Y;
                    float b = w0 * c0.Z + w1 * c1.Z + w2 * c2.Z;

                    int fbIndex = pixelIndex * 4;
                    framebuffer[fbIndex + 0] = static_cast<unsigned char>(fminf(r * 255.0f, 255.0f));
                    framebuffer[fbIndex + 1] = static_cast<unsigned char>(fminf(g * 255.0f, 255.0f));
                    framebuffer[fbIndex + 2] = static_cast<unsigned char>(fminf(b * 255.0f, 255.0f));
                    framebuffer[fbIndex + 3] = 255; // alpha
                }
            }
        }
    }
}
