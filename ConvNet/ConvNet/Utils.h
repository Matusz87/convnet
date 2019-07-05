#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include "tensor3D.h"
#include "Layer.h"
#include "Conv.h"
#include "FC.h"
#include "MaxPool.h"
#include "Softmax.h"
#include "ReLU.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

namespace utils {
	typedef std::vector<std::pair<Tensor3D<double>, Tensor3D<double>>> Dataset;

	static void PrintShape(convnet_core::Triplet& shape) {
		std::cout << "(" << shape.height << +", " << shape.width
			<< ", " << shape.depth << ")" << std::endl;
	}

	static void PrintLayerShapes(layer::Layer& layer) {
		std::cout << "Input shape: ";
		PrintShape(layer.GetInputShape());
		std::cout << "Output shape: ";
		PrintShape(layer.GetOutputShape());
		std::cout << "Gradients shape: ";
		PrintShape(layer.GetGradsShape());
	}

	static Tensor3D<double> CreateTensorFromVec(std::vector<int> vec, 
											int rows = 4, int cols = 4) {
		assert(vec.size() == rows*cols);
		cv::Mat img = cv::Mat(rows, cols, CV_32F);
		memcpy(img.data, vec.data(), vec.size() * sizeof(int));

		Tensor3D<double> tensor(rows, cols, img.channels());
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				tensor(i, j, 0) = img.at<int>(i, j);
			}
		}

		return tensor;
	}

	static Tensor3D<double> CreateTensorFromImage(std::string path) {
		cv::Mat img;
		std::string imageName(path);
		img = cv::imread(imageName.c_str(), cv::IMREAD_COLOR);
		
		std::vector<cv::Mat> bgr(3);
		cv::split(img, bgr);

		convnet_core::Tensor3D<double> tensor(img);

		return tensor;
	}

	static Tensor3D<double> CreateTensorFrom3DVec(std::vector<Tensor3D<double>> vec,
		int rows = 4, int cols = 4) {
		
		Tensor3D<double> tensor(rows, cols, vec.size());
		for (int c = 0; c < vec.size(); ++c) {
			Tensor3D<double> t = vec[c];
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					tensor(i, j, c) = t(i, j, 0);
				}
			}
		}
		

		return tensor;
	}

	static Dataset GetTrainingSet(int sample_per_class) {
		Dataset dataset;

		std::string path_dir = "../../../datasets/traffic_signs/train-52x52/";
		Tensor3D<double> X, target;

		for (int i = 0; i < 12; ++i) {
			target = Tensor3D<double>(12, 1, 1);
			target.InitZeros();
			target(i, 0, 0) = 1;
			for (int j = 0; j < sample_per_class; ++j) {
				if (((i*sample_per_class) + j) % 2000 == 0) {
					std::cout << (((i*sample_per_class) + j) / (double)(sample_per_class*12)) * 100 << "%" << std::endl;
				}
				path_dir = "../../../datasets/traffic_signs/train-52x52/";
				if (j < 10) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_000").append(std::to_string(j))
						.append(".bmp");
				}  else if (j < 100) {
					path_dir.append((std::to_string(i + 1)))
					.append("/").append(std::to_string((i + 1)))
					.append("_00").append(std::to_string(j))
					.append(".bmp");
				} else if (j < 1000) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_0").append(std::to_string(j))
						.append(".bmp");
				}
				else if (j < 4000) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_").append(std::to_string(j))
						.append(".bmp");
				}
//std::cout << path_dir << std::endl;
				X = utils::CreateTensorFromImage(path_dir);
				dataset.push_back(std::pair<Tensor3D<double>, Tensor3D<double>>(X, target));
			}
		}

		return dataset;
	}

	static Dataset GetValidationSet(int sample_per_class) {
		Dataset dataset;

		std::string path_dir = "../../../datasets/traffic_signs/train-52x52/";
		Tensor3D<double> X, target;

		for (int i = 0; i < 12; ++i) {
			target = Tensor3D<double>(12, 1, 1);
			target.InitZeros();
			target(i, 0, 0) = 1;
			for (int j = 0; j < sample_per_class; ++j) {
				path_dir = "../../../datasets/traffic_signs/train-52x52/";
				if (j < 10) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_400").append(std::to_string(j))
						.append(".bmp");
				}
				else if (j < 100) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_40").append(std::to_string(j))
						.append(".bmp");
				}
				else if (j < 500) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_4").append(std::to_string(j))
						.append(".bmp");
				}
//				std::cout << path_dir << std::endl;
				X = utils::CreateTensorFromImage(path_dir);
				dataset.push_back(std::pair<Tensor3D<double>, Tensor3D<double>>(X, target));
			}
		}

		return dataset;
	}

	static Dataset GetTestSet(int sample_per_class) {
		Dataset dataset;

		std::string path_dir = "../../../datasets/traffic_signs/train-52x52/";
		Tensor3D<double> X, target;

		for (int i = 0; i < 12; ++i) {
			target = Tensor3D<double>(12, 1, 1);
			target.InitZeros();
			target(i, 0, 0) = 1;
			for (int j = 0; j < sample_per_class; ++j) {
				path_dir = "../../../datasets/traffic_signs/train-52x52/";
				if (j < 10) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_450").append(std::to_string(j))
						.append(".bmp");
				}
				else if (j < 100) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_45").append(std::to_string(j))
						.append(".bmp");
				}
				else if (j < 500) {
					path_dir.append((std::to_string(i + 1)))
						.append("/").append(std::to_string((i + 1)))
						.append("_4").append(std::to_string(j + 500))
						.append(".bmp");
				}

				X = utils::CreateTensorFromImage(path_dir);
				dataset.push_back(std::pair<Tensor3D<double>, Tensor3D<double>>(X, target));
			}

		}

		return dataset;
	}

	static bool ComparePrediction(Tensor3D<double> pred, Tensor3D<double> target) {
		assert(pred.GetShape().height == target.GetShape().height);

		int pred_index = -1, target_index = -1;
		double max = std::numeric_limits<double>::lowest();
		for (int i = 0; i < pred.GetShape().height; ++i) {
			if (target(i, 0, 0) == 1)
				target_index = i;

			if (pred(i, 0, 0) > max) {
				pred_index = i;
				max = pred(i, 0, 0);

			}
		}

		/*std::cout << "Predicted index: " << pred_index
				  << ", target index: " << target_index << std::endl;*/

		return (pred_index == target_index);
	}

	static void WritePoolLayer(layer::MaxPool pool, std::string path) {
		std::ofstream o(path);
		nlohmann::json layer;
		
		layer["type"] = "pool";
		layer["name"] = pool.GetName();
		layer["height"] = pool.GetInputShape().height;
		layer["width"] = pool.GetInputShape().width;
		layer["depth"] = pool.GetInputShape().depth;
		layer["p_size"] = pool.GetPoolSize();
		layer["stride"] = pool.GetStride();;
		
		o << std::setw(4) << layer;
	}

	// TODO: DELETE READ FROM PATH FUNCTIONS
	static layer::MaxPool ReadPoolLayer(std::string path) {
		std::ifstream i(path);
		nlohmann::json layer;
		i >> layer;

		layer::MaxPool pool = layer::MaxPool(layer["name"], layer["height"],
			layer["width"], layer["depth"], layer["stride"], layer["p_size"]);

		return pool;
	}

	static layer::MaxPool ReadPoolLayerJSON(nlohmann::json layer) {
		layer::MaxPool pool = layer::MaxPool(layer["name"], layer["height"],
			layer["width"], layer["depth"], layer["stride"], layer["p_size"]);

		return pool;
	}

	static void WriteReLU(layer::ReLU relu, std::string path) {
		std::ofstream o(path);
		nlohmann::json layer;

		layer["type"] = "relu";
		layer["name"] = relu.GetName();
		layer["height"] = relu.GetInputShape().height;
		layer["width"] = relu.GetInputShape().width;
		layer["depth"] = relu.GetInputShape().depth;

		o << std::setw(4) << layer;
	}

	static layer::ReLU ReadReLU(std::string path) {
		std::ifstream i(path);
		nlohmann::json layer;
		i >> layer;

		layer::ReLU relu = layer::ReLU(layer["name"], layer["height"],
											 layer["width"], layer["depth"]);

		return relu;
	}

	static layer::ReLU ReadReLUJSON(nlohmann::json layer) {
		layer::ReLU relu = layer::ReLU(layer["name"], layer["height"],
			layer["width"], layer["depth"]);

		return relu;
	}

	static void WriteSoftmax(layer::Softmax softmax, std::string path) {
		std::ofstream o(path);
		nlohmann::json layer;

		layer["type"] = "softmax";
		layer["name"] = softmax.GetName();
		layer["height"] = softmax.GetInputShape().height;
		
		o << std::setw(4) << layer;
	}

	static layer::Softmax ReadSoftmax(std::string path) {
		std::ifstream i(path);
		nlohmann::json layer;
		i >> layer;

		layer::Softmax softmax = layer::Softmax(layer["name"], 
												layer["height"], 
												1, 1);

		return softmax;
	}

	static layer::Softmax ReadSoftmaxJSON(nlohmann::json layer) {
		layer::Softmax softmax = layer::Softmax(layer["name"],
												layer["height"], 
												1, 1);

		return softmax;
	}

	static void WriteConvLayer(layer::Conv conv, std::string path) {
		std::ofstream o(path);
		nlohmann::json layer;
		nlohmann::json weights;
		nlohmann::json bias;

		layer["type"] = "conv";
		layer["name"] = conv.GetName();
		layer["height"] = conv.GetInputShape().height;
		layer["width"] = conv.GetInputShape().width;
		layer["depth"] = conv.GetInputShape().depth;
		layer["f_count"] = conv.GetFilterCount();
		layer["f_size"] = conv.GetFilterSize();
		layer["stride"] = conv.GetStride();;
		layer["padding"] = conv.GetPadding();;

		for (int filter = 0; filter < conv.GetWeights().size(); ++filter) {
			nlohmann::json weight;
			for (int d = 0; d<conv.GetWeights()[filter].GetShape().depth; ++d)
				for (int h=0; h<conv.GetWeights()[filter].GetShape().height; ++h)
					for (int w = 0; w<conv.GetWeights()[filter].GetShape().width; ++w)					
						weight.push_back(conv.GetWeights()[filter](h,w,d));
			
			weights[std::to_string(filter)] = weight;
			bias.push_back(conv.GetBias()[filter](0, 0, 0));
		}
		layer["weights"] = weights;
		layer["bias"] = bias;

		o << std::setw(4) << layer;
	}

	static layer::Conv ReadConvLayer(std::string path) {
		std::ifstream i(path);
		nlohmann::json layer;
		i >> layer;
		
		layer::Conv conv = layer::Conv(layer["height"], layer["width"],
									   layer["depth"], layer["name"], 
									   layer["f_count"], layer["f_size"],
									   layer["stride"], layer["padding"]);

		int b_ind = 0;
		for (auto& element : layer["bias"]) {
			conv.GetBias()[b_ind](0, 0, 0) = element;
			++b_ind;
		}

		int filter_size = layer["f_size"];
		for (nlohmann::json::iterator it = layer["weights"].begin();
			it != layer["weights"].end(); ++it) {
			for (int h = 0; h < layer["f_size"]; ++h) {
				for (int w = 0; w < layer["f_size"]; ++w) {
					for (int d = 0; d < layer["depth"]; ++d) {
						conv.GetWeights()[std::stoi(it.key())](h, w, d) =
							it.value()[
								d * (filter_size*filter_size) +
									h * (filter_size)+
									w
							];
					}
				}
			}
		}

		return conv;
	}

	static layer::Conv ReadConvLayerJSON(nlohmann::json layer) {
		layer::Conv conv = layer::Conv(layer["height"], layer["width"],
			layer["depth"], layer["name"],
			layer["f_count"], layer["f_size"],
			layer["stride"], layer["padding"]);

		/*if (layer.count("bias") == 0 ||
			layer.count("weights") == 0) {
			return conv;
		}*/
				
		int b_ind = 0;
		for (auto& element : layer["bias"]) {
			conv.GetBias()[b_ind](0, 0, 0) = element;
			++b_ind;
		}

		int filter_size = layer["f_size"];
		for (nlohmann::json::iterator it = layer["weights"].begin();
			it != layer["weights"].end(); ++it) {
			for (int h = 0; h < layer["f_size"]; ++h) {
				for (int w = 0; w < layer["f_size"]; ++w) {
					for (int d = 0; d < layer["depth"]; ++d) {
						conv.GetWeights()[std::stoi(it.key())](h, w, d) =
							it.value()[
								d * (filter_size*filter_size) +
									h * (filter_size)+
									w
							];
					}
				}
			}
		}

		return conv;
	}

	static void WriteFCLayer(layer::FC fc, std::string path) {
		std::ofstream o(path);
		nlohmann::json layer;
		nlohmann::json weights;
		nlohmann::json bias;

		layer["type"] = "fc";
		layer["name"] = fc.GetName();
		layer["input"] = fc.GetInputShape().height;
		layer["output"] = fc.GetOutputShape().height;

		for (int inp = 0; inp < fc.GetInputShape().height; ++inp) {
			nlohmann::json weight;
			for (int out = 0; out < fc.GetOutputShape().height; ++out)
				weight.push_back(fc.GetWeights()(inp, out, 0));

			weights[std::to_string(inp)] = weight;
		}
		for (int out = 0; out<fc.GetOutputShape().height; ++out)
			bias.push_back(fc.GetBias()(out, 0, 0));

		layer["weights"] = weights;
		layer["bias"] = bias;

		o << std::setw(4) << layer;
	}

	static layer::FC ReadFCLayer(std::string path) {
		std::ifstream i(path);
		nlohmann::json layer;
		i >> layer;

		layer::FC fc = layer::FC(layer["name"], layer["input"], layer["output"]);

		int b_ind = 0;
		for (auto& element : layer["bias"]) {
			fc.GetBias()(b_ind, 0, 0) = element;
			++b_ind;
		}
		
		for (nlohmann::json::iterator it = layer["weights"].begin(); 
			it != layer["weights"].end(); ++it) {
			
			int i = 0;
			for (auto& element : it.value()) {
				fc.GetWeights()(std::stoi(it.key()), i, 0) = element;
				++i;
			}
		}

		return fc;
	}

	static layer::FC ReadFCLayerJSON(nlohmann::json layer) {
		layer::FC fc = layer::FC(layer["name"], layer["input"], layer["output"]);

		if (layer.count("bias") == 0 ||
			layer.count("weights") == 0) {
			return fc;
		}

		int b_ind = 0;
		for (auto& element : layer["bias"]) {
			fc.GetBias()(b_ind, 0, 0) = element;
			++b_ind;
		}

		for (nlohmann::json::iterator it = layer["weights"].begin();
			it != layer["weights"].end(); ++it) {

			int i = 0;
			for (auto& element : it.value()) {
				fc.GetWeights()(std::stoi(it.key()), i, 0) = element;
				++i;
			}
		}

		return fc;
	}
}