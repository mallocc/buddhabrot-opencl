#pragma once

#include <CL\cl.h>
#include <CL\opencl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <format>

std::string getOpenCLErrorString(cl_int errorCode) {
	switch (errorCode) {
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_COMPILE_PROGRAM_FAILURE:
		return "CL_COMPILE_PROGRAM_FAILURE";
	case CL_LINKER_NOT_AVAILABLE:
		return "CL_LINKER_NOT_AVAILABLE";
	case CL_LINK_PROGRAM_FAILURE:
		return "CL_LINK_PROGRAM_FAILURE";
	case CL_DEVICE_PARTITION_FAILED:
		return "CL_DEVICE_PARTITION_FAILED";
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
		return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// OpenCL 2.0 error codes
	case CL_INVALID_PROPERTY:
		return "CL_INVALID_PROPERTY";
	case CL_INVALID_IMAGE_DESCRIPTOR:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case CL_INVALID_COMPILER_OPTIONS:
		return "CL_INVALID_COMPILER_OPTIONS";
	case CL_INVALID_LINKER_OPTIONS:
		return "CL_INVALID_LINKER_OPTIONS";
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// Add more cases for other error codes as needed

	default:
		return "Unknown error";
	}
}

#define CheckError(err) __CheckError(err, __FILE__, __LINE__)

inline bool __CheckError(cl_int err, const char* file, int line) {
	if (err != CL_SUCCESS) {
		std::cerr << "OpenCL error " << err << " at " << file << ":" << line << std::endl;
		std::cout << std::format("OpenCL error {} '{}' <{}:{}>\n", err, getOpenCLErrorString(err), file, line);
		return false;
	}
	return true;
}

class CLManager;

class CLKernelArg
{
public:
	virtual bool setKernelArg(CLManager& man, int argNum) = 0;
};

class CLManager
{
public:
	cl_command_queue queue;
	cl_kernel kernel;
	cl_int error;
	cl_context context;

	cl_uint platformIdCount = 0;
	std::vector<cl_platform_id> platformIds;

	cl_uint deviceIdCount = 0;
	std::vector<cl_device_id> deviceIds;

	std::string kernelFunc;
	std::string kernelFile;

	cl_uint maxWorkItemDimensions;
	std::vector<size_t> maxWorkItemSize;

	static std::string GetPlatformName(cl_platform_id id)
	{
		size_t size = 0;
		clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

		std::string result;
		result.resize(size);
		clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
			const_cast<char*> (result.data()), nullptr);

		return result;
	}

	static std::string GetDeviceName(cl_device_id id)
	{
		size_t size = 0;
		clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

		std::string result;
		result.resize(size);
		clGetDeviceInfo(id, CL_DEVICE_NAME, size,
			const_cast<char*> (result.data()), nullptr);

		return result;
	}

	static cl_program CreateProgram(const std::string& source,
		cl_context context, cl_int* error)
	{
		size_t lengths[1] = { source.size() };
		const char* sources[1] = { source.data() };

		cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, error);

		return program;
	}

	static std::string LoadKernel(const char* name)
	{
		std::ifstream in(name);
		std::string result(
			(std::istreambuf_iterator<char>(in)),
			std::istreambuf_iterator<char>());
		return result;
	}

	bool init()
	{
		if (!CheckError(clGetPlatformIDs(0, nullptr, &platformIdCount)))
			return false;

		if (platformIdCount == 0) {
			std::cerr << "No OpenCL platform found" << std::endl;
			return false;
		}
		else {
			std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
		}

		platformIds = std::vector<cl_platform_id>(platformIdCount);
		if (!CheckError(clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr)))
			return false;

		for (cl_uint i = 0; i < platformIdCount; ++i) {
			std::cout << "\t (" << (i + 1) << ") : " << GetPlatformName(platformIds[i]) << std::endl;
		}

		// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
		if (!CheckError(clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
			&deviceIdCount)))
			return false;

		if (deviceIdCount == 0) {
			std::cerr << "No OpenCL devices found" << std::endl;
			return false;
		}
		else {
			std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
		}

		deviceIds = std::vector<cl_device_id>(deviceIdCount);
		if (!CheckError(clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount,
			deviceIds.data(), nullptr)))
			return false;

		for (cl_uint i = 0; i < deviceIdCount; ++i) {
			cl_uint info = 0;
			std::cout << "\t (" << (i + 1) << ") : " << GetDeviceName(deviceIds[i]) << std::endl;

			clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info), &info, nullptr);
			std::cout << "\t\t CL_DEVICE_MAX_COMPUTE_UNITS: " << info << std::endl;

			clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(info), &info, nullptr);
			std::cout << "\t\t CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << info << std::endl;
			maxWorkItemDimensions = info;

			clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info), &info, nullptr);
			std::cout << "\t\t CL_DEVICE_MAX_WORK_GROUP_SIZE: " << info << std::endl;

			maxWorkItemSize = std::vector<size_t>(maxWorkItemDimensions);
			clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxWorkItemDimensions, maxWorkItemSize.data(), nullptr);
			for (int j = 0; j < maxWorkItemSize.size(); ++j)
				std::cout << std::format("\t\t CL_DEVICE_MAX_WORK_GROUP_SIZE[{}]: {}\n", j, maxWorkItemSize[j]);

			cl_ulong size;
			clGetDeviceInfo(deviceIds[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
			std::cout << "\t\t CL_DEVICE_LOCAL_MEM_SIZE: " << info << std::endl;

		}

		// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
		const cl_context_properties contextProperties[] =
		{
			CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[0]),
			0, 0
		};

		cl_int error = CL_SUCCESS;
		context = clCreateContext(contextProperties, deviceIdCount,
			deviceIds.data(), nullptr, nullptr, &error);
		if (!CheckError(error))
			return false;

		std::cout << "Context created" << std::endl;

		cl_program program = CreateProgram(LoadKernel(kernelFile.c_str()),
			context, &error);
		if (!CheckError(error))
			return false;

		clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);

		// Check for build errors and get the build log
		cl_build_status buildStatus;
		if (!CheckError(clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL)))
			return false;

		if (buildStatus != CL_BUILD_SUCCESS)
		{
			// Get the build log size
			size_t buildLogSize;
			if (!CheckError(clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize)))
				return false;

			// Allocate memory for the build log
			char* buildLog = (char*)malloc(buildLogSize);

			// Get the build log
			if (!CheckError(clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL)))
				return false;

			// Output the build log (error messages)
			printf("OpenCL program build log:\n%s\n", buildLog);

			// Free allocated memory
			free(buildLog);

			return false;
		}

		kernel = clCreateKernel(program, kernelFunc.c_str(), &error);

		// Check for errors during kernel creation
		if (error != CL_SUCCESS)
		{
			// Kernel creation failed, check the error code
			std::cout << "Error creating kernel: " << error << std::endl;

			// Get the size of the build log
			size_t buildLogSize;
			if (!CheckError(clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize)))
				return false;

			// Allocate memory for the build log
			char* buildLog = (char*)malloc(buildLogSize);

			// Get the build log
			if (!CheckError(clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL)))
				return false;
			// Output the build log (error messages)
			std::cout << "Build log: " << std::endl << buildLog << std::endl;

			// Free allocated memory
			free(buildLog);

			return false;
		}
		else
		{
			// Get the function name using clGetKernelInfo
			size_t functionNameSize;
			error = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &functionNameSize);
			if (error == CL_SUCCESS)
			{
				// Allocate memory for the function name
				char* functionName = (char*)malloc(functionNameSize);

				// Get the function name
				error = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, functionNameSize, functionName, NULL);
				if (error == CL_SUCCESS)
				{
					// Output the kernel function name
					std::cout << "Kernel Function Name: " << functionName << std::endl;
				}

				// Free allocated memory
				free(functionName);
			}

			// Additional error checking for clGetKernelInfo if needed
			if (error != CL_SUCCESS)
			{
				std::cout << "Error getting kernel info: " << error << std::endl;

				return false;
			}
		}

		cl_queue_properties properties[] = {
			CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, // Example property (enable profiling)
			0 // Terminate the properties list with a zero
		};

		queue = clCreateCommandQueueWithProperties(context, deviceIds[0], properties, &error);
		if (!CheckError(error))
			return false;

		return true;
	}

	bool execute(size_t globalSize)
	{
		cl_event eve;
		const size_t globalWorkSize[] = { globalSize, 0, 0 };
		//const size_t localWorkSize[] = { maxWorkItemSize.empty() ? 1 : maxWorkItemSize[0], 0, 0 };
		const size_t localWorkSize[] = { maxWorkItemSize[0], maxWorkItemSize[1], maxWorkItemSize[2] };
		if (!CheckError(clEnqueueNDRangeKernel(queue, kernel, 1,
			nullptr,
			globalWorkSize,
			localWorkSize,
			0, nullptr, &eve)))
			return false;

		return CheckError(clFlush(queue));
	}

	bool setKernelArgs(std::vector<CLKernelArg*> args)
	{
		for (int i = 0; i < args.size(); ++i)
			if (args[i])
				if (!args[i]->setKernelArg(*this, i))
					return false;
		return true;
	}

	~CLManager()
	{
		clReleaseCommandQueue(queue);
		clReleaseKernel(kernel);
		clReleaseContext(context);
	}
};

template<typename T>
class CLBuf : public CLKernelArg
{
public:
	std::vector<T> data;
	cl_mem buf;

	CLBuf() {}

	void allocateSize(size_t size)
	{
		data = std::vector<T>(size);
	}

	bool load(CLManager& man, cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR)
	{
		buf = clCreateBuffer(man.context, flags,
			sizeof(T) * data.size(),
			data.data(), &man.error);
		return CheckError(man.error);
	}

	bool fill(CLManager& man, const T& pattern)
	{
		data = std::vector<T>(data.size(), pattern);
		return CheckError(clEnqueueFillBuffer(man.queue, buf, &pattern, sizeof(T), 0, data.size() * sizeof(T), 0, nullptr, nullptr));
	}

	bool write(CLManager& man, const std::vector<T>& newData)
	{
		data = newData;
		return write(man);
	}

	bool write(CLManager& man)
	{
		return CheckError(clEnqueueWriteBuffer(man.queue, buf, CL_TRUE, 0, sizeof(T) * data.size(), data.data(), 0, nullptr, nullptr));
	}

	bool read(CLManager& man)
	{
		return CheckError(clEnqueueReadBuffer(man.queue, buf, CL_TRUE, 0,
			sizeof(T) * data.size(),
			data.data(),
			0, nullptr, nullptr));
	}

	virtual bool setKernelArg(CLManager& man, int argNum) override
	{		
		return CheckError(clSetKernelArg(man.kernel, argNum, sizeof(cl_mem), &buf));
	}

	~CLBuf()
	{
		clReleaseMemObject(buf);
	}
};

template<typename T>
class CLVar : public CLKernelArg
{
public:
	T data;

	CLVar() {}
	CLVar(T data) : data(data) {}

	virtual bool setKernelArg(CLManager& man, int argNum) override
	{
		return CheckError(clSetKernelArg(man.kernel, argNum, sizeof(T), &data));
	}
};