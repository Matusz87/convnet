#pragma once
#include <iostream>

#include "tensor3D.h"
#include <opencv2/core/core.hpp>

class TestTensor
{
public:
	TestTensor();
	bool TestImageSplit();
	bool TestTensorFromMat();
	~TestTensor();

private:
	bool CompareMatToTensor(std::vector<cv::Mat> bgr, 
							convnet_core::Tensor3D<float> tensor);
};

