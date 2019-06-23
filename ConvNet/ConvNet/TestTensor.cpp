#include <iostream>
#include <vector>

#include "TestTensor.h"
#include "tensor3D.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>


TestTensor::TestTensor()
{
}

bool TestTensor::TestTensorFromMat() {
	std::vector<float> vec{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160};
	int rows = 4;
	int cols = 4;

	assert(vec.size() == rows*cols);
	cv::Mat img = cv::Mat(rows, cols, CV_32F); // 										
	memcpy(img.data, vec.data(), vec.size() * sizeof(float)); // change uchar to any type of data values that you want to use instead
	
	/*for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j)
			std::cout << img.at<float>(i, j) << " ";
		
		std::cout << std::endl;
	}*/
	
	std::string imageName("../../../datasets/traffic_signs/12_0000.bmp");
	img = cv::imread(imageName.c_str(), cv::IMREAD_COLOR);
	std::cout << "Image size: " << img.rows << " " << img.cols << " " << img.channels() << std::endl;

	std::vector<cv::Mat> bgr(3);   //destination array
	cv::split(img, bgr);//split source  

	/*for (int ch = 0; ch < bgr.size(); ++ch) {
		std::cout << "Channel: " << ch << std::endl;
		std::cout << "Image size: " << bgr[ch].rows << " " << bgr[ch].cols << " " << bgr[ch].channels() << std::endl;
		for (int i = 0; i < bgr[ch].rows; ++i) {
			for (int j = 0; j < bgr[ch].cols; ++j)
				std::cout << (float)bgr[ch].at<uchar>(i, j) << " ";

			std::cout << std::endl;
		}
		std::cout << std::endl;
	}*/

	convnet_core::Tensor3D<float> tensor(img);
	//PrintTensor(tensor);

	assert(CompareMatToTensor(bgr, tensor));

	cv::imshow("testMatrix", img);
	cv::waitKey(0); // Wait for a keystroke in the window

	return true;
}

bool TestTensor::TestImageSplit() {
	std::string imageName("../../../datasets/traffic_signs/1_0000.bmp");
	cv::Mat image;
	image = cv::imread(imageName.c_str(), cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::namedWindow("Display window", cv::WINDOW_KEEPRATIO); // Create a window for display.
	imshow("Display window", image);                // Show our image inside it.

	std::vector<cv::Mat> bgr(3);   //destination array
	cv::split(image, bgr);//split source  

	std::cout << bgr[0].size() << " " << bgr[1].size() << bgr[2].size() << std::endl;

	imshow("blue.png", bgr[0]); //blue channel
	imshow("green.png", bgr[1]); //green channel
	imshow("red.png", bgr[2]); //red channel

	cv::Mat merged;
	cv::merge(bgr, merged);
	imshow("merged", merged); //red channel

	cv::waitKey(0); // Wait for a keystroke in the window
	return true;
}

TestTensor::

TestTensor::~TestTensor()
{
}

bool TestTensor::CompareMatToTensor(std::vector<cv::Mat> bgr, 
									convnet_core::Tensor3D<float> tensor)
{
	bool equals = true;
	for (int ch = 0; ch < bgr.size(); ++ch) {
		std::cout << "Channel: " << ch << std::endl;
		std::cout << "Image size: " << bgr[ch].rows << " " << bgr[ch].cols << " " << bgr[ch].channels() << std::endl;
		for (int i = 0; i < bgr[ch].rows; ++i) {
			for (int j = 0; j < bgr[ch].cols; ++j)
				if ((float)bgr[ch].at<uchar>(i, j) != tensor.get(i, j, ch)) {
					std::cout << "Error: " << (float)bgr[ch].at<uchar>(i, j) << " != "
					<< tensor.get(i, j, ch) << std::endl;

					return false;
				}
		}
	}
	std::cout << "Mat and Tensor3D are equals" << std::endl;

	return equals;
}
