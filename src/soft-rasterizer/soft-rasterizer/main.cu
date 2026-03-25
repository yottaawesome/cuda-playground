#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <climits>

#include "math.cuh"
#include "rasterizer.cuh"

// ── CUDA helpers ────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess)                                                   \
            throw std::runtime_error(std::string(#call) + " failed: "             \
                                     + cudaGetErrorString(err));                   \
    } while (0)

template<typename T>
auto CudaMalloc(size_t count) -> T*
{
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
auto CudaUpload(T* devicePtr, const T* hostPtr, size_t count) -> void
{
    CUDA_CHECK(cudaMemcpy(devicePtr, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
auto CudaDownload(T* hostPtr, const T* devicePtr, size_t count) -> void
{
    CUDA_CHECK(cudaMemcpy(hostPtr, devicePtr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// ── PPM writer ──────────────────────────────────────────────────────────────

auto WritePPM(const std::string& filename,
              const unsigned char* rgba,
              int width, int height) -> void
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open " + filename + " for writing");

    // PPM header (P6 = binary RGB)
    file << "P6\n" << width << " " << height << "\n255\n";

    // Write RGB (skip alpha)
    for (int i = 0; i < width * height; i++)
    {
        file.put(static_cast<char>(rgba[i * 4 + 0]));
        file.put(static_cast<char>(rgba[i * 4 + 1]));
        file.put(static_cast<char>(rgba[i * 4 + 2]));
    }
}

// ── Scene definition ────────────────────────────────────────────────────────

auto MakeTriangleVertices() -> std::vector<Rasterizer::Vertex>
{
    // A single triangle with red, green, blue vertices
    return {
        { { 0.0f,  0.8f,  0.0f }, { 1.0f, 0.0f, 0.0f } },  // top    — red
        { {-0.8f, -0.6f,  0.0f }, { 0.0f, 1.0f, 0.0f } },  // left   — green
        { { 0.8f, -0.6f,  0.0f }, { 0.0f, 0.0f, 1.0f } },  // right  — blue
    };
}

auto MakeCubeVertices() -> std::vector<Rasterizer::Vertex>
{
    // A unit cube made of 12 triangles (2 per face), centered at origin.
    // Each face gets a distinct color so the 3D shape is visible.
    Math::Vec3 ftl = { -0.5f,  0.5f,  0.5f };  // front-top-left
    Math::Vec3 ftr = {  0.5f,  0.5f,  0.5f };
    Math::Vec3 fbl = { -0.5f, -0.5f,  0.5f };
    Math::Vec3 fbr = {  0.5f, -0.5f,  0.5f };
    Math::Vec3 btl = { -0.5f,  0.5f, -0.5f };
    Math::Vec3 btr = {  0.5f,  0.5f, -0.5f };
    Math::Vec3 bbl = { -0.5f, -0.5f, -0.5f };
    Math::Vec3 bbr = {  0.5f, -0.5f, -0.5f };

    Math::Vec3 red    = { 1.0f, 0.2f, 0.2f };
    Math::Vec3 green  = { 0.2f, 1.0f, 0.2f };
    Math::Vec3 blue   = { 0.3f, 0.3f, 1.0f };
    Math::Vec3 yellow = { 1.0f, 1.0f, 0.2f };
    Math::Vec3 cyan   = { 0.2f, 1.0f, 1.0f };
    Math::Vec3 purple = { 1.0f, 0.2f, 1.0f };

    return {
        // Front face (red)
        { ftl, red }, { fbl, red }, { fbr, red },
        { ftl, red }, { fbr, red }, { ftr, red },
        // Back face (green)
        { btr, green }, { bbr, green }, { bbl, green },
        { btr, green }, { bbl, green }, { btl, green },
        // Left face (blue)
        { btl, blue }, { bbl, blue }, { fbl, blue },
        { btl, blue }, { fbl, blue }, { ftl, blue },
        // Right face (yellow)
        { ftr, yellow }, { fbr, yellow }, { bbr, yellow },
        { ftr, yellow }, { bbr, yellow }, { btr, yellow },
        // Top face (cyan)
        { btl, cyan }, { ftl, cyan }, { ftr, cyan },
        { btl, cyan }, { ftr, cyan }, { btr, cyan },
        // Bottom face (purple)
        { bbl, purple }, { bbr, purple }, { fbr, purple },
        { bbl, purple }, { fbr, purple }, { fbl, purple },
    };
}

auto BuildMVP(float rotationAngleDeg, float aspect) -> Math::Mat4
{
    float angle = rotationAngleDeg * 3.14159265f / 180.0f;

    Math::Mat4 model = Math::RotationY(angle);
    Math::Mat4 view  = Math::LookAt(
        { 0.0f, 0.5f, 2.5f },   // eye
        { 0.0f, 0.0f, 0.0f },   // target
        { 0.0f, 1.0f, 0.0f }    // up
    );
    Math::Mat4 proj = Math::Perspective(
        45.0f * 3.14159265f / 180.0f,  // 45° FOV
        aspect,
        0.1f,   // near
        100.0f  // far
    );

    return proj * view * model;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    constexpr int Width  = 800;
    constexpr int Height = 600;
    constexpr int PixelCount = Width * Height;

    // Parse optional rotation angle (default: 30°)
    float rotationDeg = 30.0f;
    std::string mesh = "cube";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--angle" && i + 1 < argc)
            rotationDeg = std::stof(argv[++i]);
        else if (arg == "--mesh" && i + 1 < argc)
            mesh = argv[++i];
    }

    // Select geometry
    std::vector<Rasterizer::Vertex> vertices;
    if (mesh == "triangle")
        vertices = MakeTriangleVertices();
    else
        vertices = MakeCubeVertices();

    int vertexCount   = static_cast<int>(vertices.size());
    int triangleCount = vertexCount / 3;

    std::cout << "Rendering " << mesh << " (" << triangleCount << " triangles) "
              << "at " << Width << "x" << Height
              << ", rotation = " << rotationDeg << " degrees\n";

    Math::Mat4 mvp = BuildMVP(rotationDeg, static_cast<float>(Width) / Height);

    // ── Allocate device memory ──────────────────────────────────────────
    auto* dVertices   = CudaMalloc<Rasterizer::Vertex>(vertexCount);
    auto* dScreenPos  = CudaMalloc<Math::Vec3>(vertexCount);
    auto* dFramebuf   = CudaMalloc<unsigned char>(PixelCount * 4);
    auto* dDepthBuf   = CudaMalloc<int>(PixelCount);

    // Upload vertex data
    CudaUpload(dVertices, vertices.data(), vertexCount);

    // Clear framebuffer to dark gray (25, 25, 25) and depth to INT_MAX
    {
        std::vector<unsigned char> clearColor(PixelCount * 4);
        for (int i = 0; i < PixelCount; i++)
        {
            clearColor[i * 4 + 0] = 25;
            clearColor[i * 4 + 1] = 25;
            clearColor[i * 4 + 2] = 25;
            clearColor[i * 4 + 3] = 255;
        }
        CudaUpload(dFramebuf, clearColor.data(), PixelCount * 4);

        std::vector<int> clearDepth(PixelCount, INT_MAX);
        CudaUpload(dDepthBuf, clearDepth.data(), PixelCount);
    }

    // ── Launch vertex shader ────────────────────────────────────────────
    {
        int threadsPerBlock = 256;
        int blocks = (vertexCount + threadsPerBlock - 1) / threadsPerBlock;
        Rasterizer::VertexShader<<<blocks, threadsPerBlock>>>(
            dVertices, dScreenPos, vertexCount, mvp, Width, Height
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Launch rasterizer ───────────────────────────────────────────────
    {
        dim3 blockSize(16, 16);
        dim3 gridSize(
            (Width  + blockSize.x - 1) / blockSize.x,
            (Height + blockSize.y - 1) / blockSize.y
        );
        Rasterizer::Rasterize<<<gridSize, blockSize>>>(
            dVertices, dScreenPos, triangleCount,
            dFramebuf, dDepthBuf, Width, Height
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Read back and save ──────────────────────────────────────────────
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> hostFramebuf(PixelCount * 4);
    CudaDownload(hostFramebuf.data(), dFramebuf, PixelCount * 4);

    std::string filename = "output.ppm";
    WritePPM(filename, hostFramebuf.data(), Width, Height);
    std::cout << "Wrote " << filename << "\n";

    // ── Cleanup ─────────────────────────────────────────────────────────
    cudaFree(dVertices);
    cudaFree(dScreenPos);
    cudaFree(dFramebuf);
    cudaFree(dDepthBuf);

    return 0;
}
