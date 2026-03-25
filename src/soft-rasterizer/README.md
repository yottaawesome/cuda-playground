# Soft Rasterizer

A software rasterizer built entirely on CUDA compute kernels — no OpenGL, DirectX, or Vulkan. The GPU does all the work (vertex transformation, rasterization, shading) and the host just sets up data and writes the result to an image file.

This is a learning project. The goal is to demystify what a GPU graphics pipeline does by implementing one from scratch using only general-purpose compute.

## AI usage disclosure

This was developed by Github Copilot in Claude Opus 4.6 mode at my prompting. The purpose of this was to have a reference CUDA-based software rasterizer that I can use for learning. The original idea came to me when I thinking about 3D graphics and CUDA, and then it occurred to me you can use CUDA to offload all the linear algebra calculations.

## Building

Requires Visual Studio 2026+ with the C++ workload and the NVIDIA CUDA Toolkit (13.2+).

```powershell
msbuild soft-rasterizer.slnx /p:Configuration=Debug /p:Platform=x64
```

Or open `soft-rasterizer.slnx` in Visual Studio and build with **Ctrl+B**.

## Running

```powershell
# Render a rotated cube (default)
.\x64\Debug\soft-rasterizer.exe

# Render a flat RGB triangle
.\x64\Debug\soft-rasterizer.exe --mesh triangle --angle 0

# Render the cube at 45° rotation
.\x64\Debug\soft-rasterizer.exe --mesh cube --angle 45
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--mesh <name>` | `cube` | Mesh to render. `cube` (12 triangles, 6 colored faces) or `triangle` (1 triangle, RGB vertices). |
| `--angle <degrees>` | `30` | Y-axis rotation angle in degrees. |

### Output

Writes `output.ppm` in the working directory. PPM is a simple uncompressed image format — most image viewers and editors can open it (including IrfanView, GIMP, and VS Code with extensions). The resolution is fixed at **800×600**.

## Architecture

The rasterizer follows the same conceptual pipeline as a real GPU, but implemented as two CUDA kernels:

```
                     Host (CPU)                              Device (GPU)
              ─────────────────────                 ───────────────────────────

              Define vertex data                    Kernel 1 — VertexShader
              Build Model/View/Projection     ───►    One thread per vertex
              Allocate device buffers                 Multiply position by MVP
              Upload vertices + MVP                   Perspective divide → NDC
                                                      Viewport transform → pixels

                                                    Kernel 2 — Rasterize
                                                      One thread per pixel
                                              ───►    For each triangle:
                                                        Bounding box early-out
                                                        Edge function coverage test
                                                        Barycentric interpolation
                                                        atomicMin depth test
                                                        Write color to framebuffer

              Read back framebuffer           ◄───
              Write PPM file
```

### File structure

```
soft-rasterizer/
  soft-rasterizer.slnx              Solution file
  soft-rasterizer/
    soft-rasterizer.vcxproj          Project file (CUDA 13.2 build customizations)
    main.cu                          Host entry point, scene setup, PPM output
    math.cuh                         Vector/matrix math (Vec3, Vec4, Mat4)
    rasterizer.cuh                   GPU kernels (VertexShader, Rasterize)
```

## How it works

### Vertex shader (`rasterizer.cuh`)

One CUDA thread per vertex. Each thread:

1. **Multiplies** the vertex position by the MVP (Model × View × Projection) matrix to get **clip-space** coordinates.
2. **Perspective divides** (divides XYZ by W) to get **NDC** (Normalized Device Coordinates), where X and Y are in [-1, 1] and Z is in [0, 1].
3. **Viewport transforms** NDC to screen pixel coordinates. Y is flipped so that +Y in world space points up in the image.

### Rasterizer + fragment shader (`rasterizer.cuh`)

One CUDA thread per pixel. Each thread loops over all triangles and for each one:

1. **Bounding box test** — skips the triangle if the pixel is outside its screen-space bounding box.
2. **Edge function test** — evaluates three edge functions to determine if the pixel center lies inside the triangle. Accepts both CW and CCW winding (the viewport Y-flip reverses winding, so both must be handled).
3. **Barycentric interpolation** — the three edge function values, once normalized, give the barycentric coordinates (w0, w1, w2). These are used to interpolate per-vertex colors and depth.
4. **Depth test** — the interpolated depth is reinterpreted as an integer via `__float_as_int` and compared against the depth buffer using `atomicMin`. Since IEEE 754 positive floats sort the same as integers, this gives correct closest-fragment-wins behavior without locks.
5. **Framebuffer write** — if the fragment wins the depth test, its interpolated color is written to the RGBA framebuffer.

### Math library (`math.cuh`)

All types are marked `__host__ __device__` so they work on both CPU and GPU:

- **`Vec3`** / **`Vec4`** — basic vector types with arithmetic operators, dot product, cross product, and normalize.
- **`Mat4`** — 4×4 column-major matrix with matrix-matrix and matrix-vector multiplication.
- **`RotationY`** — builds a Y-axis rotation matrix.
- **`LookAt`** — right-handed look-at view matrix.
- **`Perspective`** — right-handed perspective projection matrix with depth mapped to [0, 1].

### MVP matrix (`main.cu`)

The Model-View-Projection matrix is built on the host and passed by value to the vertex shader kernel:

- **Model** — `RotationY(angle)`, rotates the mesh around the Y axis.
- **View** — `LookAt(eye=(0, 0.5, 2.5), target=origin, up=Y)`, a camera slightly above and in front of the scene.
- **Projection** — `Perspective(45° FOV, 800/600 aspect, near=0.1, far=100)`.

## Points of interest

### No graphics API at all

This project doesn't link against any graphics library. It uses only CUDA runtime (`cudaMalloc`, `cudaMemcpy`, kernel launches) and standard C++. The output is a raw image file.

### Depth buffer via `atomicMin`

Real GPUs have dedicated hardware for the depth test (ROPs). Here, multiple threads may try to write the same pixel simultaneously if triangles overlap. The solution is `atomicMin` on an integer depth buffer — a single atomic operation that both tests and updates the depth. The trick of reinterpreting a float as an int via `__float_as_int` works because IEEE 754 positive floats have the same ordering as their integer bit patterns.

### Winding-order invariant rasterization

The viewport transform flips Y (so +Y is up in world space but down in pixel coordinates). This reverses the winding order of triangles. Rather than requiring a specific winding convention, the rasterizer accepts both by checking if all three edge functions are ≥ 0 **or** all ≤ 0. The barycentric math works either way because dividing negative weights by a negative area produces positive coordinates.

### Brute-force pixel-parallel approach

Each pixel thread loops over every triangle. This is O(pixels × triangles) — correct but slow for large scenes. Real rasterizers use tile-based binning to limit which triangles each pixel group tests. This is a natural next optimization to try.

## Possible extensions

- **Texture mapping** — upload a texture to device memory, use barycentric-interpolated UVs to sample it in the fragment shader.
- **Lighting** — add per-vertex normals, interpolate them, and compute diffuse/specular shading.
- **Triangle clipping** — clip triangles against the near/far planes before rasterization.
- **Tile-based binning** — assign triangles to screen tiles to reduce per-pixel work.
- **CUDA–OpenGL interop** — write directly to an OpenGL PBO and display in a window for real-time rendering without the host readback.
- **Perspective-correct interpolation** — divide vertex attributes by W before interpolation and correct afterward (required for textures to look right on perspective-foreshortened surfaces).
