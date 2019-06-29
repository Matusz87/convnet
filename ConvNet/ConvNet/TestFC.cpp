#include "TestFC.h"
#include "FC.h"
#include "tensor3D.h"
#include "Utils.h"

TestFC::TestFC() { }
TestFC::~TestFC() { }

bool TestFC::TestConstructor() {
	std::cout << "TestFC::TestConstructor" << std::endl;

	Tensor3D<double> tensor(28, 28, 1);
	layer::FC fc(tensor, "fc", 128);
	utils::PrintLayerShapes(fc);

	return true;
}

bool TestFC::TestForward() {
	std::cout << "TestFC::TestForward" << std::endl;

	std::vector<int> vec({ 1,2, 3 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 3, 1);

	std::vector<int> w_vec({ 1,2,3,4,5,6 });
	Tensor3D<double> weights = utils::CreateTensorFromVec(w_vec, 3, 2);

	layer::FC fc(input, "fc", 2);

	fc.GetWeights() = weights;

	fc.Forward(fc.GetInput());
	std::cout << "Result: " << std::endl;
	convnet_core::PrintTensor(fc.GetOutput());

	assert(fc.GetOutput()(0, 0, 0) == 22 && fc.GetOutput()(1, 0, 0) == 28);

	return true;
}

bool TestFC::TestBackprop() {
	std::cout << "TestFC::TestBackprop" << std::endl;

	std::vector<int> vec({ 1,2, 3 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 3, 1);

	std::vector<int> w_vec({ 1,2,3,4,5,6 });
	Tensor3D<double> weights = utils::CreateTensorFromVec(w_vec, 3, 2);

	std::vector<int> e_vec({ 3,2 });
	Tensor3D<double> error = utils::CreateTensorFromVec(e_vec, 2, 1);

	//Tensor3D<double> tensor(2, 1, 1);
	layer::FC fc(input, "fc", 2);

	fc.GetWeights() = weights;

	fc.Forward(fc.GetInput());
	std::cout << "Result: " << std::endl;
	convnet_core::PrintTensor(fc.GetOutput());

	std::cout << "Backprop: " << std::endl;
	fc.Backprop(error);
	std::cout << "Input gradient: " << std::endl;
	convnet_core::PrintTensor(fc.GetGradInput()); 
	std::cout << "Weight gradient: " << std::endl;
	convnet_core::PrintTensor(fc.GetGradWeights());

	assert(fc.GetOutput()(0, 0, 0) == 22 && fc.GetOutput()(1, 0, 0) == 28);

	return true;
}
