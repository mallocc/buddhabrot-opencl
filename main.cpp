// OpenCL Test 4.cpp : Defines the entry point for the console application.
//

#include "BuddhabrotRenderer.h"
#include "Timer.h"

#include <stack>
#include "Log.h"

int main(int argc, char* argv[])
{
	srand(static_cast<unsigned int>(time(NULL)));

	LOG("Program Entry");

	std::stack<std::string> args;
	for (int a = argc - 1; a >= 1; --a)
		args.push(argv[a]);

	for (int a = 0; a < argc; ++a)
		std::cout << argv[a] << " ";
	std::cout << std::endl;

	TimerMS timer;
	timer.start();

	BuddhabrotRenderer bb;
	BuddhabrotRenderer::Stage stage;

	bool success = true;
	// read argumetions and populate options
	while (!args.empty() && success)
	{
		std::string arg(args.top());
		args.pop();

		if (!arg.empty())
		{
			// a flag!
			if (arg[0] == '-')
			{
				std::string option(arg.substr(1));

				if (!option.empty())
				{
					if (option[0] == '-')
						option = option.substr(1);

					auto checkAndSet = [&](std::function<void(const std::string&)> callback)
					{
						if (!args.empty())
						{
							callback(args.top());
							args.pop();
						}
						else
						{
							LOG("No option value supplied: " << arg);
							success = false;
						}
					};

					auto checkAndSetAndReturn = [&](std::function<int(const std::string&)> callback)
					{
						if (!args.empty())
						{
							success = callback(args.top());
							if (!success)
								LOG("Invalid option value supplied: " << args.top());
							args.pop();
						}
						else
						{
							LOG("No option value supplied: " << arg);
							success = false;
						}
					};

					//auto checkSetVolumeVals = [&](int& val)
					//{
					//	checkAndSetAndReturn([&](const std::string& in) -> bool
					//		{
					//			if (!bb.dimensions.contains(in))
					//				return false;
					//			val = bb.dimensions[in];
					//			return true;
					//		});
					//};

					if (option == "w" || option == "width")
						checkAndSet([&](const std::string& in) { bb.width = std::stoi(in); });
					else if (option == "h" || option == "height")
						checkAndSet([&](const std::string& in) { bb.height = std::stoi(in); });

					//else if (option == "vax" || option == "volume-a-x")
					//	checkSetVolumeVals(bb.volumeAX);
					//else if (option == "vay" || option == "volume-a-y")
					//	checkSetVolumeVals(bb.volumeAY);
					//else if (option == "vaz" || option == "volume-a-z")
					//	checkSetVolumeVals(bb.volumeAZ);
					//else if (option == "vbx" || option == "volume-b-x")
					//	checkSetVolumeVals(bb.volumeBX);
					//else if (option == "vby" || option == "volume-b-y")
					//	checkSetVolumeVals(bb.volumeBY);
					//else if (option == "vbz" || option == "volume-b-z")
					//	checkSetVolumeVals(bb.volumeBZ);

					else if (option == "i" || option == "iterations")
						checkAndSet([&](const std::string& in) { stage.iterations = std::stoi(in); });
					else if (option == "ir" || option == "iterations-red")
						checkAndSet([&](const std::string& in) { stage.iterationsR = std::stoi(in); });
					else if (option == "ig" || option == "iterations-green")
						checkAndSet([&](const std::string& in) { stage.iterationsG = std::stoi(in); });
					else if (option == "ib" || option == "iterations-blue")
						checkAndSet([&](const std::string& in) { stage.iterationsB = std::stoi(in); });
					else if (option == "im" || option == "iterations-min")
						checkAndSet([&](const std::string& in) { stage.iterationsMin = std::stoi(in); });

					else if (option == "gamma")
						checkAndSet([&](const std::string& in) { stage.gamma = std::stof(in); });
					else if (option == "radius")
						checkAndSet([&](const std::string& in) { bb.radius = std::stof(in); });

					else if (option == "re0" || option == "x0" || option == "real0")
						checkAndSet([&](const std::string& in) { stage.v0.re = std::stof(in); });
					else if (option == "im0" || option == "y0" || option == "imaginary0")
						checkAndSet([&](const std::string& in) { stage.v0.im = std::stof(in); });

					else if (option == "re1" || option == "x1" || option == "real1")
						checkAndSet([&](const std::string& in) { stage.v1.re = std::stof(in); });
					else if (option == "im1" || option == "y1" || option == "imaginary1")
						checkAndSet([&](const std::string& in) { stage.v1.im = std::stof(in); });

					else if (option == "s" || option == "samples")
						checkAndSet([&](const std::string& in) { stage.samples = std::stoll(in); });
					else if (option == "o" || option == "output")
						checkAndSet([&](const std::string& in) { bb.filename = in; });
					else if (option == "steps")
						checkAndSet([&](const std::string& in) { stage.steps = std::stoi(in); });
					
					else if (option == "alpha" || option == "a")
						checkAndSet([&](const std::string& in) { stage.alpha = std::stof(in); });
					else if (option == "beta" || option == "b")
						checkAndSet([&](const std::string& in) { stage.beta = std::stof(in); });
					else if (option == "theta" || option == "t")
						checkAndSet([&](const std::string& in) { stage.theta = std::stof(in); });
					else if (option == "phi" || option == "p")
						checkAndSet([&](const std::string& in) { stage.phi = std::stof(in); });

					else if (option == "zScalerA" || option == "zsa" || option == "z.a.scaler")
						checkAndSet([&](const std::string& in) { stage.zScalerA = std::stof(in); });
					else if (option == "zScalerB" || option == "zsb" || option == "z.b.scaler")
						checkAndSet([&](const std::string& in) { stage.zScalerB = std::stof(in); });
					else if (option == "zScalerC" || option == "zsc" || option == "z.c.scaler")
						checkAndSet([&](const std::string& in) { stage.zScalerC = std::stof(in); });

					else if (option == "zAngleA" || option == "zaa" || option == "z.a.angle")
						checkAndSet([&](const std::string& in) { stage.zAngleA = std::stof(in); });
					else if (option == "zAngleB" || option == "zab" || option == "z.b.angle")
						checkAndSet([&](const std::string& in) { stage.zAngleB = std::stof(in); });
					else if (option == "zAngleC" || option == "zac" || option == "z.c.angle")
						checkAndSet([&](const std::string& in) { stage.zAngleC = std::stof(in); });

					else if (option == "zYScaleA" || option == "zysa" || option == "z.a.yscale")
						checkAndSet([&](const std::string& in) { stage.zYScaleA = std::stof(in); });
					else if (option == "zYScaleB" || option == "zysb" || option == "z.b.yscale")
						checkAndSet([&](const std::string& in) { stage.zYScaleB = std::stof(in); });
					else if (option == "zYScaleC" || option == "zysc" || option == "z.c.yscale")
						checkAndSet([&](const std::string& in) { stage.zYScaleC = std::stof(in); });

					else if (option == "mhRatio")
						checkAndSet([&](const std::string& in) { stage.mhRatio = std::stof(in); });

					//else if (option == "escape-trajectories" || option == "et")
					//	checkAndSet([&](const std::string& in) { bb.escapeThreshold = std::stoi(in); });
					//else if (option == "escape-trajectories-red" || option == "etr")
					//	checkAndSet([&](const std::string& in) { bb.escapeThresholdR = std::stoi(in); });
					//else if (option == "escape-trajectories-green" || option == "etg")
					//	checkAndSet([&](const std::string& in) { bb.escapeThresholdG = std::stoi(in); });
					//else if (option == "escape-trajectories-blue" || option == "etb")
					//	checkAndSet([&](const std::string& in) { bb.escapeThresholdB = std::stoi(in); });

					else if (option == "threads" || option == "j" || option == "jobs")
						checkAndSet([&](const std::string& in) { bb.jobs = std::stoi(in); });

					else if (option == "counter-offset")
						checkAndSet([&](const std::string& in) { bb.counterOffset = std::stoi(in); });
					else if (option == "bezier-enable")
						stage.bezier = true;
					else if (option == "bezier-disable")
						stage.bezier = false;
					else if (option == "gen-in-region")
						bb.generateOnlyInRegion = true;

					else if (option == "throttle-factor")
						checkAndSet([&](const std::string& in) { bb.throttleFactor = std::stof(in); });

					else if (option == "silent")
						bb.silent = true;

					else if (option == "next" || option == "next-stage" || option == "n")
					{
						bb.stages.push_back(stage);
						stage = {};
					}
					else if (option == "next-cpy" || option == "next-stage-copy" || option == "nc")
						bb.stages.push_back(stage);
					else
					{
						LOG("Unknown option: " << arg);
						success = false;
					}
				}
				else
				{
					LOG("Invaid option: " << arg);
					success = false;
				}
			}
			else
			{
				LOG("Invaid argument: " << arg);
				success = false;
			}
		}
		else
		{
			LOG("Empty argument.");
			success = false;
		}
	}

	if (!success)
	{
		LOG("Program exit early.");
		return 1;
	}

	bb.stages.push_back(stage);

	bb.init();

	if (!bb.initCL())
		return 1;

	if (!bb.run())
		return 1;

	timer.stop();

	LOG(std::format("Total Time: {}", timer.getAverageTimestamp()));

	return 0;
}




