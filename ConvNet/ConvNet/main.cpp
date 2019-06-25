#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "TestTensor.h"
#include "TestReLU.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
	int i;
	TestReLU testReLU;
	testReLU.TestConstructor();
	testReLU.TestConstructorWithTensor();
	testReLU.TestConstructorWithMat();
	testReLU.TestForward();

	std::cin >> i;

	//TestTensor testTensor;
	//test.TestImageSplit();
	//testTensor.TestTensorFromMatIntSuccess();
	//testTensor.TestTensorFromMatSuccess();
	//test.TestTensorFromMatIntFail();
	//test.TestTensorFromMatFail();

	

	return 0;
}