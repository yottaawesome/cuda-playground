# CUDA & GPU Graphics Pipeline — Notes

A collection of notes from building a software rasterizer in CUDA and exploring GPU graphics concepts.

## Can you build a graphics API with just CUDA compute?

Yes — this is a well-trodden path and a great learning exercise. The core idea is a **software rasterizer on GPU compute**.

**What works well:**
- Vertex transformations (model/view/projection) are trivially parallel — one thread per vertex, matrix multiplies via shared memory
- You'd write to a framebuffer in global memory, then copy it back to host (or display via CUDA-OpenGL interop)

**Where it gets interesting:**
- **Rasterization** — you need to implement triangle setup, edge functions, and scanline/tile-based filling yourself in kernel code
- **Depth buffering** — requires `atomicMin` on a Z-buffer since multiple triangles may hit the same pixel
- **Interpolation** — barycentric coords for vertex attributes (color, UVs, normals)

**What you lose vs. fixed-function HW:**
- The GPU's dedicated rasterizer units, texture samplers, ROPs, and hardware depth test are *extremely* fast and power-efficient — a compute-only path won't match them
- But for learning GPU programming and linear algebra, it's fantastic

**Rough architecture:**
```
Host: load mesh, build MVP matrix, upload to GPU
Kernel 1: vertex shader — transform vertices (parallel per-vertex)
Kernel 2: rasterizer — for each triangle, emit fragments
Kernel 3: fragment shader — shade each pixel, atomic write to framebuffer + Z-buffer
Host: read back framebuffer, display (or use CUDA-GL interop to skip the copy)
```

---

## The soft rasterizer we built

See `src/soft-rasterizer/README.md` for full documentation. Key files:

| File | Purpose |
|---|---|
| `math.cuh` | `Vec3`, `Vec4`, `Mat4` with `__host__ __device__` — rotation, lookAt, perspective projection |
| `rasterizer.cuh` | Two GPU kernels: **VertexShader** (per-vertex MVP + viewport transform) and **Rasterize** (per-pixel edge function test, barycentric interpolation, `atomicMin` depth buffer) |
| `main.cu` | Host code: defines geometry (triangle or cube), builds MVP, allocates GPU buffers, launches kernels, writes PPM |

**Usage:**
```powershell
soft-rasterizer.exe --mesh cube --angle 45    # rotated cube
soft-rasterizer.exe --mesh triangle --angle 0  # flat RGB triangle
```

---

## Will more GPU power be exposed to overcome software rasterizer limitations?

**It's already happening**, and the trend strongly suggests more.

**The trajectory:**
- OpenGL hid nearly everything behind a state machine
- DirectX 12 / Vulkan exposed command buffers, memory management, pipeline barriers — work the driver used to do silently
- **Mesh shaders** (DX12 Ultimate / Vulkan) already replaced the fixed-function vertex-fetch and tessellation stages with general compute-like programs
- **Work graphs** (DX12) let the GPU dispatch its own work without CPU round-trips

**Where it's heading:**
The fixed-function units being "protected" are really just **rasterization** (triangle setup + scan conversion) and **ROPs** (depth test + blend + framebuffer write). These exist as dedicated silicon because they're extremely fast and power-efficient at what they do. But:

1. **Rasterization is already optional** — ray tracing (RT cores) is a parallel path that bypasses the rasterizer entirely, and it's getting faster each generation
2. **Software rasterization on compute is viable at scale** — Nanite in Unreal Engine 5 does exactly this for small triangles, falling back to hardware rasterization only for large ones, because the fixed-function rasterizer actually has overhead per-triangle that exceeds the cost of doing it in a compute shader for tiny triangles
3. **The ROPs are the last holdout** — `atomicMin` depth testing (what we did) works but is slower than dedicated hardware. However, as atomic throughput improves each GPU generation, this gap narrows

**The likely endgame:**
Not that fixed-function hardware disappears, but that the API exposes it as an **optional accelerator** you can choose to use — like RT cores or tensor cores today. You'd write your pipeline in compute by default and opt into hardware rasterization or hardware depth testing when it benefits you, rather than being forced through a rigid pipeline.

---

## Performance gap: `atomicMin` vs. dedicated hardware depth testing

**The dedicated ROP depth test** operates at the full pixel fill rate — on a modern GPU like an RTX 4080, that's ~200+ billion depth tests/second. It runs in specialized silicon right next to the framebuffer memory, with its own cache, doing the read-compare-write in a single cycle per pixel. It also batches operations over 2×2 quad fragments and benefits from hierarchical Z (Hi-Z) that can reject entire tiles of pixels before testing individual ones.

**`atomicMin` on global memory** has fundamentally different characteristics:
- Each atomic is a read-modify-write to L2 cache or VRAM — typically ~10-100× slower than a dedicated ROP operation per pixel
- Contention is the killer: when many triangles overlap the same pixel, threads serialize on that address. ROPs are designed for exactly this access pattern; the L2 atomic path is not
- No Hi-Z equivalent — every fragment must actually execute the atomic, while ROPs can skip large pixel regions early

**In practice**, the gap varies dramatically by workload:

| Scenario | Gap |
|---|---|
| Large triangles, heavy overdraw (many layers hitting same pixel) | **10–50×** slower — atomic contention dominates |
| Medium triangles, moderate overdraw | **3–10×** slower |
| Tiny triangles (sub-pixel), low overdraw | **~1–2×** or even **faster** — the fixed-function rasterizer has per-triangle setup cost that exceeds the work, which is exactly why Nanite software-rasterizes small triangles |

The Nanite insight is the key nuance: the performance gap isn't a fixed number — it depends on triangle size. As geometry gets denser (which is the trend), the fixed-function advantage shrinks and can even invert.

---

## D3D12 Work Graphs

Work graphs solve a fundamental bottleneck: **the CPU telling the GPU what to do next**.

### The problem

In a traditional frame, the CPU records a command list — "dispatch this compute shader, then dispatch that one, then draw these triangles" — and submits it to the GPU. If the second dispatch depends on what the first one *produced* (e.g., how many triangles survived a culling pass), you have two bad options:

1. **Read back to CPU** — GPU finishes pass 1, result copied to CPU, CPU decides what to dispatch, submits pass 2. Massive latency (the pipeline stalls).
2. **Over-allocate** — dispatch the worst-case amount of work for pass 2 regardless of what pass 1 actually produced. Wastes GPU cycles.

### What work graphs do

A work graph is a **DAG of GPU shader nodes** where each node can **spawn work for other nodes**, all on the GPU with zero CPU involvement.

```
┌─────────────┐
│  Node A      │  "Cull triangles"
│  (Compute)   │
└──────┬───────┘
       │ produces N records
       ▼
┌─────────────┐
│  Node B      │  "Rasterize surviving triangles"
│  (Compute)   │
└──────┬───────┘
       │ some triangles need subdivision
       ▼
┌─────────────┐
│  Node C      │  "Subdivide and re-rasterize"
│  (Compute)   │
└─────────────┘
```

Node A runs, and as each thread finishes, it can emit **work records** — small structs that get routed to Node B's input queue. The GPU scheduler launches Node B threads as records arrive, with exactly the right amount of work. Node B can in turn emit records to Node C, or even back to itself (recursion).

### Key properties

- **Data-driven dispatch** — the amount of work at each stage is determined by what the previous stage actually produced, not by a CPU-side guess
- **No round-trips** — the entire multi-pass pipeline runs on the GPU in a single `DispatchGraph` call from the CPU
- **Dynamic topology** — nodes can feed back into themselves, enabling recursive algorithms (LOD refinement, BVH construction, adaptive tessellation) that previously required awkward CPU-side loops
- **The GPU becomes self-scheduling** — it's essentially a task graph executor where shaders enqueue work for other shaders

### Current state

- **D3D12**: shipped with Agility SDK. Microsoft's [DirectX-Graphics-Samples](https://github.com/microsoft/DirectX-Graphics-Samples) repo has `D3D12WorkGraphs` samples.
- **Vulkan**: no ratified extension yet. `VK_AMDX_shader_enqueue` exists as an AMD vendor extension (experimental). Vulkan typically lags D3D12 on bleeding-edge features by 1–2 years.

---

## CUDA host–GPU synchronization

CUDA has a layered synchronization model, from coarsest to finest:

### 1. Full device sync — `cudaDeviceSynchronize()`

The CPU blocks until **all** GPU work completes. Simple but brutal — no overlap between CPU and GPU.

### 2. Stream-level sync — CUDA streams + events

A **stream** is an ordered queue of GPU work (kernel launches, memcpys). Work within a stream executes in order; work across different streams can overlap.

```cpp
cudaStream_t streamA, streamB;
cudaStreamCreate(&streamA);
cudaStreamCreate(&streamB);

kernelA<<<..., streamA>>>();  // these two can run
kernelB<<<..., streamB>>>();  // concurrently on the GPU

cudaStreamSynchronize(streamA);  // CPU blocks until only streamA is done
// streamB might still be running
```

**Events** are the closest thing to semaphores — they're GPU-side timestamps you can record and wait on:

```cpp
cudaEvent_t event;
cudaEventCreate(&event);

kernelA<<<..., streamA>>>();
cudaEventRecord(event, streamA);     // GPU records event after kernelA

cudaStreamWaitEvent(streamB, event); // streamB pauses until event fires
kernelB<<<..., streamB>>>();         // safe to read kernelA's output
```

This is **GPU→GPU** synchronization — no CPU involvement. The CPU just sets up the dependency graph and walks away.

### 3. CPU polling — `cudaEventQuery()`

Non-blocking check from the CPU:

```cpp
cudaEventRecord(event, stream);
// ... do CPU work ...
if (cudaEventQuery(event) == cudaSuccess) {
    // GPU is done — read results
}
```

This lets you overlap CPU work with GPU work without blocking.

### 4. Within-kernel sync

- **`__syncthreads()`** — barrier for all threads in a **block** (shared memory fence)
- **`__threadfence()`** — memory fence visible to all blocks (global memory)
- **Cooperative groups** — flexible sub-block and grid-wide synchronization (grid sync requires the cooperative launch API)

### Summary table

| Mechanism | Scope | Blocking? | Use case |
|---|---|---|---|
| `cudaDeviceSynchronize` | All GPU work | CPU blocks | Simple programs, debugging |
| `cudaStreamSynchronize` | One stream | CPU blocks | Wait for specific pipeline stage |
| `cudaEvent` + `StreamWaitEvent` | Stream→stream | GPU waits, CPU free | Pipeline stages, multi-stream overlap |
| `cudaEventQuery` | CPU polling | Non-blocking | Overlap CPU + GPU work |
| `__syncthreads` | Block-local | Threads wait | Shared memory coordination |
| `__threadfence` | Grid-wide | Memory fence | Cross-block communication |

Events don't have a counter like traditional semaphores, but they signal completion of a point in a stream, which is exactly the primitive you need to build dependency chains across concurrent GPU work.

---

## Skill development roadmap

A progression building on the soft rasterizer, from immediate to long-term:

### Extend the soft rasterizer

1. **Perspective-correct texture mapping** — add UV coordinates to vertices, upload a texture as a device buffer, sample it in the fragment shader. Hit the classic perspective-correctness bug (affine interpolation looks wrong on angled surfaces) and fix it by dividing attributes by W.

2. **Phong lighting** — add per-vertex normals, interpolate them, compute diffuse + specular against a point light in the fragment shader.

3. **Triangle clipping** — implement Sutherland-Hodgman clipping against the near plane in a kernel between the vertex shader and rasterizer.

4. **Tile-based binning** — replace the brute-force "every pixel tests every triangle" with a two-pass approach: first bin triangles into screen tiles, then each tile only tests its assigned triangles. This is how real mobile GPUs work.

### Parallel algorithm projects

5. **Parallel radix sort** — sort a million integers on the GPU. Teaches scan (prefix sum), shared memory bank conflicts, and multi-pass kernel design.

6. **Parallel BVH construction** — build a bounding volume hierarchy on the GPU using Morton codes + radix sort. The data structure that makes ray tracing fast.

### Ray tracing

7. **CUDA path tracer** — cast rays from the camera, bounce them around a scene, accumulate color. Start with spheres, then add triangles + BVH.

8. **Hybrid renderer** — rasterize primary visibility (what we already have), then switch to ray tracing for shadows and reflections only.

### Suggested order

```
You are here
     │
     ├── 1. Texture mapping
     ├── 2. Lighting
     ├── 3. Clipping
     │
     ├── 5. Radix sort
     ├── 4. Tile binning
     │
     ├── 7. Path tracer
     ├── 6. GPU BVH
     │
     └── 8. Hybrid renderer
```

### Resources

- **"Ray Tracing in One Weekend"** (Peter Shirley) — free online, easily adapted to CUDA
- **"Physically Based Rendering: From Theory to Implementation"** (pbrt) — the bible, dense but authoritative
- **NVIDIA's CUDA samples** on GitHub — specifically `simpleTexture`, `reduction`, `scan`, `sortingNetworks`
- **"A trip through the Graphics Pipeline"** (Fabian Giesen's blog series) — explains what the hardware actually does at each stage
