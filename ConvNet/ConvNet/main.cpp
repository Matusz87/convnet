#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "TestTensor.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
	TestTensor test;
	//test.TestImageSplit();
	test.TestTensorFromMat();

	return 0;
}