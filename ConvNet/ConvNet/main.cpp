#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "TestTensor.h"
#include "TestReLU.h"
#include "TestMaxPool.h"
#include "TestNet.h"
#include "TestConv.h"
#include "TestFC.h"
#include "TestSoftmax.h"

#include "MaxPool.h"
#include "Conv.h"
#include "ReLU.h"
#include "FC.h"
#include "Softmax.h"
#include "Model.h"
#include "Utils.h"

#include <iostream>
#include <string>
#include <memory>

int main(int argc, char** argv)
{
	int i;

	TestFC testfc;
	/*fc.TestConstructor();
	fc.TestForward();
	fc.TestBackprop();*/

	TestNet testNet;
	/*testNet.TestReluPool();*/
	//testNet.TrainConvLayer();
	//testNet.TrainFC2();
	//testNet.TrainMNISTdigit();
	//testNet.TrainSignFC();
	//testNet.TrainSign();
//	testNet.TrainSignCE2();
	//testNet.Evaluate();

	TestTensor t;
	/*t.TestInitZeros();
	t.TestInitRandom();*/

	TestConv testConv;
	//testConv.TestUnpadding();
	/*testConv.TestConstructor();
	testConv.TestPadding();
	testConv.TestForward();
	testConv.TestForward2();
	*/
	//testConv.TestForward();
	//testConv.TestForward3();
	//testConv.TestBackprop();
	//testConv.TestBackprop2();
	//testConv.TestBackpropPadded();
	/*testConv.TestForward3();	
	testConv.TestForwardDeep();
	testConv.TestForwardPadded();
*/
	TestMaxPool testMaxPool;
	//testMaxPool.TestConstructor();
	//testMaxPool.TestConstructorWithTensor();
	//testMaxPool.TestConstructorWithMat();
	/*testMaxPool.TestForward();
	testMaxPool.TestForwardWithMat();
	testMaxPool.TestBackprop();
*/
/*	TestReLU testReLU;
	testReLU.TestConstructor();
	testReLU.TestConstructorWithTensor();
	testReLU.TestConstructorWithMat();
	testReLU.TestForward();
*/

	TestSoftmax testSoftmax;
	//testSoftmax.TestForward();

	// Create layers.
	layer::MaxPool pool_0("pool_0", 52, 52, 3, 2, 2);
	pool_0 = utils::ReadPoolLayer("models/pool_0.json");
	layer::Conv conv(26, 26, 3, "conv", 12, 3, 1, 0);
	conv = utils::ReadConvLayer("models/1200/conv-9-.json");
	layer::ReLU relu("relu", 24, 24, 12);
	relu = utils::ReadReLU("models/relu.json");
	layer::MaxPool pool("pool", 24, 24, 12, 2, 2);
	pool = utils::ReadPoolLayer("models/pool.json");
	layer::Conv conv_2(12, 12, 12, "conv_2", 8, 3, 1, 0);
	conv_2 = utils::ReadConvLayer("models/1200/conv_2-9-.json");
	layer::ReLU relu_1("relu_1", 10, 10, 8);
	relu_1 = utils::ReadReLU("models/relu_1.json");
	layer::FC fc("fc", 10 * 10 * 8, 64);
	fc = utils::ReadFCLayer("models/1200/fc-9-.json");
	layer::ReLU relu_2("relu_2", 64, 1, 1);
	relu_2 = utils::ReadReLU("models/relu_2.json");
	layer::FC fc_2("fc_2", 64, 12);
	fc_2 = utils::ReadFCLayer("models/1200/fc_2-9-.json");
	layer::Softmax softmax("softmax", 12, 1, 1);
	softmax = utils::ReadSoftmax("models/softmax.json");

	convnet_core::Model model;
	model.Add(pool_0);
	model.Add(conv);
	model.Add(relu);
	model.Add(pool);
	model.Add(conv_2);
	model.Add(relu_1);
	model.Add(fc);
	model.Add(relu_2);
	model.Add(fc_2);
	model.Add(softmax);

	utils::Dataset testSet = utils::GetTestSet(10);
	Tensor3D<double> input, target, predicted;
	int correct = 0;
	for (int m = 0; m < testSet.size(); ++m) {
		if (m % 100 == 0) {
			std::cout << "... (" << m << "/" << testSet.size() << ")";
			std::cout << "Test set accuracy: " << correct / (double)testSet.size() << std::endl;
		}

		input = testSet[m].first;
		target = testSet[m].second;

		predicted = model.Predict(input);

		if (utils::ComparePrediction(predicted, target))
			++correct;
	}

	std::cout << std::endl;
	std::cout << "Final Test set accuracy: " << correct / (double)testSet.size() << std::endl;

	std::cin >> i;

	//TestTensor testTensor;
	//test.TestImageSplit();
	//testTensor.TestTensorFromMatIntSuccess();
	//testTensor.TestTensorFromMatSuccess();
	//test.TestTensorFromMatIntFail();
	//test.TestTensorFromMatFail();

	

	return 0;
}