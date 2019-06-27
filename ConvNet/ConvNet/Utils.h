#pragma once
#include <iostream>
#include "tensor3D.h"
#include "Layer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace utils {

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
		std::cout << "Image size: " << img.rows << " " << img.cols << " " << img.channels() << std::endl;

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
}