#pragma once
#include <iostream>

#include "tensor3D.h"
#include <opencv2/core/core.hpp>

class TestTensor
{
public:
	TestTensor();
	bool TestTensorFromMatIntSuccess();
	bool TestTensorFromMatIntFail();
	bool TestTensorFromMatSuccess();
	bool TestTensorFromMatFail();
	bool TestInitZeros();
	bool TestInitRandom();
	~TestTensor();

private:
	bool CompareMatToTensor(std::vector<cv::Mat> bgr, 
							convnet_core::Tensor3D<double> tensor);
	bool CompareMatIntToTensor(cv::Mat img,
							   convnet_core::Tensor3D<double> tensor);
};

