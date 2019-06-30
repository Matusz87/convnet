#include "TestSoftmax.h"
#include "Softmax.h"
#include "Utils.h"

TestSoftmax::TestSoftmax() { }
TestSoftmax::~TestSoftmax() { }

void TestSoftmax::TestForward() {
	std::cout << "TestSoftmax::TestForward" << std::endl;
	convnet_core::Tensor3D<double> tensor(4, 1, 1);
	tensor(0, 0, 0) = -3.44; tensor(1, 0, 0) = 1.16;
	tensor(2, 0, 0) = -0.81; tensor(3, 0, 0) = 3.91;
	convnet_core::Triplet shape = tensor.GetShape();
	
	layer::Softmax softmax(tensor, "softmax");
	convnet_core::PrintTensor(softmax.GetInput());
	utils::PrintLayerShapes(softmax);

	std::cout << "Forward: " << std::endl;
	softmax.Forward(tensor);
	convnet_core::PrintTensor(softmax.GetOutput());

	std::cout << "Loss: " << std::endl;
	//convnet_core::PrintTensor(softmax.Loss());
}
