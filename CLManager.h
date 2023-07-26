#pragma once

#include <CL\cl.h>
#include <CL\opencl.h>
#include <vector>
#include <format>


class CLManager;

#define CheckError(err) CLManager::__CheckError(err, __FILE__, __LINE__)

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

	static std::string getOpenCLErrorString(cl_int errorCode);

	static bool __CheckError(cl_int err, const char* file, int line);
	
	static std::string GetPlatformName(cl_platform_id id);

	static std::string GetDeviceName(cl_device_id id);

	static cl_program CreateProgram(const std::string& source,
		cl_context context, cl_int* error);

	static std::string LoadKernel(const char* name);

	bool init();

	bool execute(size_t globalSize);

	bool setKernelArgs(std::vector<CLKernelArg*> args);

	~CLManager();
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