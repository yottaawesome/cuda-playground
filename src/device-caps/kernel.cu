#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <iostream>
#include <vector>

namespace Cuda
{
	struct Error : std::runtime_error
	{
		Error(cudaError_t error) : std::runtime_error(cudaGetErrorString(error)) {}
	};

	struct Result
	{
		cudaError_t error;
		Result(cudaError_t error) : error(error) {}
		operator cudaError_t() const { return error; }
		operator bool() const { return error == cudaSuccess; }
	};

	namespace Device
	{
		auto GetCount() -> int
		{
			int deviceCount = 0;
			if (!Result{ cudaGetDeviceCount(&deviceCount) })
				throw Error(cudaGetLastError());
			return deviceCount;
		}

		auto GetProperties(int device) -> cudaDeviceProp
		{
			cudaDeviceProp properties;
			if (!Result{ cudaGetDeviceProperties(&properties, device) })
				throw Error(cudaGetLastError());
			return properties;
		}

		auto Enumerate() -> std::vector<cudaDeviceProp>
		{
			auto devices = std::vector<cudaDeviceProp>{};
			auto deviceCount = int{ GetCount() };
			devices.reserve(deviceCount);
			for (int i = 0; i < deviceCount; ++i)
				devices.push_back(GetProperties(i));
			return devices;
		}
	}
}

int main()
try
{
	auto devices = Cuda::Device::Enumerate();
	for (const auto& device : devices)
		std::cout << "Device: " << device.name << ", Compute Capability: " << device.major << "." << device.minor << std::endl;

    return 0;
}
catch (const std::exception& e)
{
	std::cerr << "Error: " << e.what() << std::endl;
	return 1;
}

