#include <iostream>
#include <exception>

import greet;

namespace Cuda
{
	struct CudaResult
	{
		cudaError_t Value = static_cast<cudaError_t>(0);
		constexpr auto IsSuccess() const noexcept -> bool { return Value == cudaSuccess; }
		auto GetErrorString() const noexcept -> const char* { return cudaGetErrorString(Value); }
		constexpr auto IsError() const noexcept -> bool { return Value != cudaSuccess; }
		constexpr operator bool() const noexcept { return IsSuccess(); }
	};

	struct CudaDeleter
	{
		static auto operator()(void* devicePtr) -> void
		{
			if (auto result = cudaFree(devicePtr); result != cudaSuccess)
				throw std::runtime_error("Failed to free memory on the GPU: " + std::string(cudaGetErrorString(result)));
		}
	};
	template<typename T>
	using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

	template<typename T>
	auto Malloc() -> T*
	{
		T* devicePtr;
		if (auto result = cudaMalloc(&devicePtr, sizeof(T)); result != cudaSuccess)
			throw std::runtime_error("Failed to allocate memory on the GPU: " + std::string(cudaGetErrorString(result)));
		return devicePtr;
	}

	auto Free(void* devicePtr) -> void
	{
		if (auto result = cudaFree(devicePtr); result != cudaSuccess)
			throw std::runtime_error("Failed to free memory on the GPU: " + std::string(cudaGetErrorString(result)));
	}

	auto Delete(void* ptr) -> void
	{
		ptr ? Free(ptr) : void();
	}

	template<typename T>
	auto CopyDeviceToHost(T* hostPtr, const T* devicePtr) -> void
	{
		if (auto result = cudaMemcpy(hostPtr, devicePtr, sizeof(T), cudaMemcpyDeviceToHost); result != cudaSuccess)
			throw std::runtime_error("Failed to copy memory from the GPU: " + std::string(cudaGetErrorString(result)));
	}

	template<typename T>
	auto CopyDeviceToHost(const T* devicePtr) -> T
	{
		T hostValue;
		if (auto result = cudaMemcpy(&hostValue, devicePtr, sizeof(T), cudaMemcpyDeviceToHost); result != cudaSuccess)
			throw std::runtime_error("Failed to copy memory from the GPU: " + std::string(cudaGetErrorString(result)));
		return hostValue;
	}
}

// run on the GPU
namespace QuickMath
{
	/* 
	* nvcc qualifiers:
	* - __global__ — Declares a kernel function. Runs on the device (GPU), callable from host (or device with dynamic parallelism). Must return void.
	* - __device__ — Runs on the device, callable only from device code (other __device__ or __global__ functions). Can return values.
	* - __host__ — Runs on the host (CPU). This is the default for all functions, so it's usually omitted unless combined with __device__.
	*/
	__global__
	void Add(int a, int b, int* c)
	{
		*c = a + b;
	}
}

int main()
{
	// This will hold the result of the addition on the GPU. The memory lives on the GPU, and can only be accessed by the GPU.
	int* deviceResult = Cuda::Malloc<int>();

	// Add and store the result in deviceResult. This runs on the GPU. Note that the runtime handles the argument passing from the host to the device.
	QuickMath::Add<<<1, 1>>>(2, 3, deviceResult);

	// Copy the result back to the host (CPU) memory so we can print it. This is synchronous, so it will block until the copy is complete.
	int hostResult = Cuda::CopyDeviceToHost(deviceResult);

	// Print the result. This runs on the CPU.
	std::cout<< "The result of 2 + 3 is: " << hostResult << std::endl;

	// Free the memory on the GPU.
	Cuda::Free(deviceResult);

	return 0;
}

