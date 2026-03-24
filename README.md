# CUDA playground

A playground for experimenting with NVIDIA CUDA on Windows using Visual Studio.

## Prerequisites

- **Visual Studio 2026** (or later) with the **Desktop development with C++** workload
- **NVIDIA CUDA Toolkit** (e.g. 13.2) — download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- An NVIDIA GPU with CUDA support

During installation the CUDA Toolkit will register its build customizations with Visual Studio automatically. You can verify this by checking that `CUDA <version>.props` and `CUDA <version>.targets` exist in your VS build customizations directory (e.g. `C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\`).

## Setting up a CUDA project from scratch

### 1. Create a C++ Console Application

In Visual Studio, create a new **Console App** (C++) project. This gives you a standard `.vcxproj` with MSVC tooling.

### 2. Rename source files from `.cpp` to `.cu`

CUDA source files **must** use the `.cu` extension so that `nvcc` (the NVIDIA CUDA compiler) processes them instead of the standard MSVC compiler.

In Solution Explorer, rename your source file (e.g. `main.cpp` → `main.cu`).

### 3. Add CUDA build customizations to the project

This is the key step. You need to tell MSBuild to use the CUDA compiler for `.cu` files.

Right-click the project in Solution Explorer → **Build Dependencies** → **Build Customizations…** and check the box for your CUDA version (e.g. **CUDA 13.2**).

This adds two imports to your `.vcxproj`:

```xml
<!-- Near the top, inside ExtensionSettings -->
<ImportGroup Label="ExtensionSettings">
  <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.2.props" />
</ImportGroup>

<!-- Near the bottom, inside ExtensionTargets -->
<ImportGroup Label="ExtensionTargets">
  <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.2.targets" />
</ImportGroup>
```

### 4. Set the file's Item Type to CUDA C/C++

Right-click your `.cu` file in Solution Explorer → **Properties** → **General** → set **Item Type** to **CUDA C/C++**.

This changes the item in the `.vcxproj` from `<ClCompile>` to `<CudaCompile>`:

```xml
<!-- Before (standard C++) -->
<ItemGroup>
  <ClCompile Include="main.cpp" />
</ItemGroup>

<!-- After (CUDA) -->
<ItemGroup>
  <CudaCompile Include="main.cu" />
</ItemGroup>
```

### 5. Write CUDA code and build

You can now use CUDA-specific syntax (`__global__`, `__device__`, `<<<...>>>` launch syntax, etc.) in your `.cu` files. Standard C++ code works alongside it.

```cpp
#include <iostream>

__global__ void hello_world()
{
    printf("Hello World from GPU!\n");
}

int main()
{
    hello_world<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Build with **Ctrl+B** or from the command line:

```powershell
msbuild hello-world.slnx /p:Configuration=Debug /p:Platform=x64
```

## Using C++20 modules with CUDA

C++20 modules can be used for **host-side code** in a CUDA project. Module interface files (`.ixx`) are compiled by MSVC, and `.cu` files can `import` those modules. Device code (`__global__`, `__device__`) must remain in `.cu` files — modules can only contain host code.

This works because `nvcc` splits `.cu` files into device and host code. The `import` declarations are handled by the CUDA frontend (for parsing) and then by the MSVC host compiler (for actual module resolution).

### Enabling C++20 in the CUDA compiler

Add `--std=c++20` to the `CudaCompile` additional options for each configuration:

```xml
<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
  <!-- ... -->
  <CudaCompile>
    <AdditionalOptions>--std=c++20 %(AdditionalOptions)</AdditionalOptions>
  </CudaCompile>
</ItemDefinitionGroup>
```

### Adding a module interface file

Add your `.ixx` file to the project as a regular `ClCompile` item (MSVC handles it, not nvcc). Use the **global module fragment** for standard library includes:

```cpp
// greet.ixx
module;
#include <iostream>        // use #include in the global module fragment
export module greet;

export void greet_from_host() {
    std::cout << "Hello from host module!\n";
}
```

Then import it from your `.cu` file:

```cpp
// main.cu
#include <cstdio>
import greet;

__global__ void hello_kernel() {
    printf("Hello from GPU!\n");
}

int main() {
    greet_from_host();
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Required `.vcxproj` fixes

Getting this to actually build requires two fixes in the `.vcxproj` to work around the way the CUDA build customization interacts with MSVC module compilation.

#### Fix 1: Build ordering — compile modules before CUDA files

By default, the CUDA build targets run **before** `ClCompile`, so the `.ifc` (compiled module interface) file doesn't exist yet when `nvcc` processes your `.cu` file. Override the scheduling so CUDA compilation happens after `ClCompile`:

```xml
<ImportGroup Label="ExtensionSettings">
  <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.2.props" />
</ImportGroup>

<!-- Add this PropertyGroup right after ExtensionSettings -->
<PropertyGroup>
  <CudaCompileAfterTargets>ClCompile</CudaCompileAfterTargets>
  <CudaCompileBeforeTargets>Link</CudaCompileBeforeTargets>
</PropertyGroup>
```

#### Fix 2: `.ifc` file name and location

`nvcc`'s CUDA frontend (cudafe++, based on the EDG parser) resolves `import` declarations by searching for `.ifc` files **only in the current working directory** (the project directory during MSBuild). It does not respect MSVC's `/ifcSearchDir` flag or `-I` include paths for this purpose.

Meanwhile, MSBuild's default module compilation names the `.ifc` file after the **source file** (e.g. `test.ixx.ifc`), not the **module name** (e.g. `greet.ifc`), and places it in the intermediate output directory (e.g. `x64\Debug\`). This means nvcc can't find it.

The fix is to override `ModuleOutputFile` on each `.ixx` item to output the `.ifc` to the project directory, named after the module:

```xml
<ItemGroup>
  <ClCompile Include="greet.ixx">
    <ModuleOutputFile>.\greet.ifc</ModuleOutputFile>
  </ClCompile>
</ItemGroup>
```

Since this places generated `.ifc` files in the project directory, add `*.ifc` to your `.gitignore`.

### C5050 warnings

You may see warnings like:

```
warning C5050: Possible incompatible environment while importing module 'greet':
  _M_FP_PRECISE is defined in module command line and not in current command line
```

These are benign — they result from minor preprocessor definition differences between how MSVC compiled the module and how `nvcc` invokes the host compiler. Suppress them if desired with:

```xml
<CudaCompile>
  <AdditionalOptions>--std=c++20 -Xcompiler /wd5050 %(AdditionalOptions)</AdditionalOptions>
</CudaCompile>
```

### Key limitations

- **Device code cannot live in modules** — `__global__` and `__device__` functions must be in `.cu` files
- **`nvcc` only finds `.ifc` files in the project directory** — the CUDA frontend does not support `/ifcSearchDir`, `-I` paths, or the `IFCPATH` environment variable for module lookup
- **Each `.ixx` file needs an explicit `ModuleOutputFile`** mapping in the `.vcxproj` so the `.ifc` is named after the module, not the source file
- **Use `#include` in the global module fragment**, not `import <header>`, for standard library headers — header units are not supported through `nvcc`

## Project structure

```
src/
  hello-world/
    hello-world.slnx          # Solution file
    hello-world/
      hello-world.vcxproj     # Project file (with CUDA build customizations)
      main.cu                 # CUDA source (imports modules, contains device code)
      test.ixx                # C++20 module interface (host-only code)
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `__global__` is not recognized | File is `.cpp`, not `.cu` | Rename to `.cu` and set Item Type to **CUDA C/C++** |
| `nvcc` not invoked at all | Missing CUDA build customizations | Add via **Build Dependencies → Build Customizations** |
| `Cannot open include file: 'cuda_runtime.h'` | CUDA Toolkit not installed or `CUDA_PATH` not set | Reinstall the CUDA Toolkit |
| Linker errors for `cudaDeviceSynchronize` etc. | CUDA runtime not linked | Ensure CUDA build customizations are enabled (they handle linking automatically) |
| `could not find module file for module "X"` from nvcc | `.ifc` not in project dir, or wrong filename | Set `<ModuleOutputFile>.\X.ifc</ModuleOutputFile>` on the `.ixx` item |
| `could not find module file` but `.ifc` exists | CUDA targets run before `ClCompile` | Add `<CudaCompileAfterTargets>ClCompile</CudaCompileAfterTargets>` (see above) |
| `Stray '"' character` from nvcc | Trailing `\` in path creates `\"` | Avoid trailing backslashes in `-Xcompiler` quoted arguments; use `$(IntDir.TrimEnd('\'))` |
| C5050 "Possible incompatible environment" warnings | Preprocessor defs differ between module and CUDA compilation | Benign; suppress with `-Xcompiler /wd5050` |
