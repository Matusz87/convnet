#pragma once
#include <iostream>
#include <algorithm>

#include "tensor3D.h"
#include "Layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

	static Dataset GetTrainingSet() {
		Dataset dataset;

		std::string path_dir = "../../../datasets/traffic_signs/train-52x52/";
		Tensor3D<double> X, target;

		for (int i = 0; i < 12; ++i) {
			target = Tensor3D<double>(12, 1, 1);
			target.InitZeros();
			target(i, 0, 0) = 1;
			for (int j = 0; j < 5; ++j) {
				path_dir = "../../../datasets/traffic_signs/train-52x52/";
				path_dir.append((std::to_string(i + 1)))
					.append("/").append(std::to_string((i + 1)))
					.append("_000").append(std::to_string(j))
					.append(".bmp");

				X = utils::CreateTensorFromImage(path_dir);
				dataset.push_back(std::pair<Tensor3D<double>, Tensor3D<double>>(X, target));
			}
		}

		return dataset;
	}

	static Dataset GetValidationSet() {
		Dataset dataset;

		std::string path_dir = "../../../datasets/traffic_signs/train-52x52/";
		Tensor3D<double> X, target;

		for (int i = 0; i < 2; ++i) {
			target = Tensor3D<double>(3, 1, 1);
			target.InitZeros();
			target(i, 0, 0) = 1;
			for (int j = 0; j < 9; ++j) {
				path_dir = "../../../datasets/traffic_signs/train-52x52/";
				path_dir.append((std::to_string(i + 1)))
					.append("/").append(std::to_string((i + 1)))
					.append("_100").append(std::to_string(j))
					.append(".bmp");

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
				  << ", target index: " << target_index << std::endl;
*/
		return (pred_index == target_index);
	}
}