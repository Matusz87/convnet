#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "TestTensor.h"
#include "TestReLU.h"
#include "TestMaxPool.h"
#include "TestNet.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
	int i;
	TestNet testNet;
	testNet.TestReluPool();

	/*TestMaxPool testMaxPool;
	testMaxPool.TestConstructor();
	testMaxPool.TestConstructorWithTensor();
	testMaxPool.TestConstructorWithMat();
	testMaxPool.TestForward();
	testMaxPool.TestForwardWithMat();
	testMaxPool.TestBackprop();

	TestReLU testReLU;
	testReLU.TestConstructor();
	testReLU.TestConstructorWithTensor();
	testReLU.TestConstructorWithMat();
	testReLU.TestForward();
*/
	std::cin >> i;

	//TestTensor testTensor;
	//test.TestImageSplit();
	//testTensor.TestTensorFromMatIntSuccess();
	//testTensor.TestTensorFromMatSuccess();
	//test.TestTensorFromMatIntFail();
	//test.TestTensorFromMatFail();

	

	return 0;
}