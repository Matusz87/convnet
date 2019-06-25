#include <iostream>
#include <vector>

#include "TestTensor.h"
#include "tensor3D.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>


TestTensor::TestTensor() { }

bool TestTensor::TestTensorFromMatSuccess() {
	cv::Mat img;
	std::string imageName("../../../datasets/traffic_signs/1_0000_.bmp");
	img = cv::imread(imageName.c_str(), cv::IMREAD_COLOR);
	std::cout << "Image size: " << img.rows << " " << img.cols << " " << img.channels() << std::endl;

	std::vector<cv::Mat> bgr(3);
	cv::split(img, bgr);

	//for (int ch = 0; ch < bgr.size(); ++ch) {
	//	std::cout << "Channel: " << ch << std::endl;
	//	std::cout << "Image size: " << bgr[ch].rows << " " << bgr[ch].cols << " " << bgr[ch].channels() << std::endl;
	//	for (int i = 0; i < bgr[ch].rows; ++i) {
	//		for (int j = 0; j < bgr[ch].cols; ++j)
	//			std::cout << (double)bgr[ch].at<uchar>(i, j) << " ";

	//		std::cout << std::endl;
	//	}
	//	std::cout << std::endl;
	//}

	convnet_core::Tensor3D<double> tensor(img);
	//PrintTensor(tensor);

	assert(CompareMatToTensor(bgr, tensor));

	cv::imshow("testMatrix", img);
	cv::waitKey(0); 

	return true;
}

bool TestTensor::TestTensorFromMatFail() {
	std::string imageName("../../../datasets/traffic_signs/1_0000.bmp");
	std::string imageName2("../../../datasets/traffic_signs/2_0000.bmp");
	cv::Mat img, img2;
	img = cv::imread(imageName.c_str(), cv::IMREAD_COLOR);
	img2 = cv::imread(imageName2.c_str(), cv::IMREAD_COLOR);

	std::cout << "Image size: " << img.rows << " " << img.cols << " " << img.channels() << std::endl;

	std::vector<cv::Mat> bgr(3);
	cv::split(img, bgr);

	convnet_core::Tensor3D<double> tensor(img2);
	//PrintTensor(tensor);

	assert(CompareMatToTensor(bgr, tensor));

	return true;
}

bool TestTensor::TestTensorFromMatIntSuccess() {
	std::vector<int> vec{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160 };
	int rows = 4;
	int cols = 4;

	assert(vec.size() == rows*cols);
	cv::Mat img = cv::Mat(rows, cols, CV_32F);									
	memcpy(img.data, vec.data(), vec.size() * sizeof(int));

	std::cout << "Mat: " << std::endl;
	convnet_core::Tensor3D<double> tensor(rows, cols, img.channels());
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << img.at<int>(i, j) << " ";
			tensor(i, j, 0) = img.at<int>(i, j);
		}

		std::cout << std::endl;
	}

	PrintTensor(tensor);

	assert(CompareMatIntToTensor(img, tensor));

	return true;
}

bool TestTensor::TestTensorFromMatIntFail()
{
	std::vector<int> vec{ 10, 20, 30, 40, 50, 60, 70, 80, 90 };
	int rows = 3;
	int cols = 3;

	assert(vec.size() == rows*cols);
	cv::Mat img = cv::Mat(rows, cols, CV_32F);
	memcpy(img.data, vec.data(), vec.size() * sizeof(int));

	std::cout << "Mat: " << std::endl;
	convnet_core::Tensor3D<double> tensor(rows, cols, img.channels());
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << img.at<int>(i, j) << " ";
			tensor(i, j, 0) = img.at<int>(i, j)+1;
		}

		std::cout << std::endl;
	}

	PrintTensor(tensor);

	assert(CompareMatIntToTensor(img, tensor));

	return true;
}

TestTensor::

TestTensor::~TestTensor()
{
}

bool TestTensor::CompareMatToTensor(std::vector<cv::Mat> bgr, 
									convnet_core::Tensor3D<double> tensor)
{
	bool equals = true;
	for (int ch = 0; ch < bgr.size(); ++ch) {
		std::cout << "Channel: " << ch << std::endl;
		std::cout << "Image size: " << bgr[ch].rows << " " 
				  << bgr[ch].cols << " " << bgr[ch].channels() << std::endl;

		for (int i = 0; i < bgr[ch].rows; ++i) {
			for (int j = 0; j < bgr[ch].cols; ++j)
				// Take into account the normalization factor.
				if ((double)bgr[ch].at<uchar>(i, j) != tensor.get(i, j, ch)*255.0) {
					std::cout << "Error: " << (double)bgr[ch].at<uchar>(i, j) << " != "
							  << tensor.get(i, j, ch)*255.0 << std::endl;

					return false;
				}
		}
	}
	std::cout << "Mat and Tensor3D are equal." << std::endl << std::endl;

	return equals;
}

bool TestTensor::CompareMatIntToTensor(cv::Mat img, convnet_core::Tensor3D<double> tensor)
{
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j)
			if ((double)img.at<int>(i, j) != tensor.get(i, j, 0)) {
				std::cout << "Error: " << (double)img.at<int>(i, j) << " != "
					<< tensor.get(i, j, 0) << std::endl;

				return false;
			}
	}
	std::cout << "Mat<int> and Tensor3D are equal." << std::endl << std::endl;

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