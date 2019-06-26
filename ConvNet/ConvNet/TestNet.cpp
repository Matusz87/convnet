#include "TestNet.h"

#include <iostream>
#include "Utils.h"
#include "tensor3D.h"
#include "Layer.h"
#include "ReLU.h"
#include "MaxPool.h"

bool TestNet::TestReluPool() {
	std::vector<int> vec{ 10, -20, 30, 40, -50, 60, 70, -80, 90, 100, -110, 120, 130, -140, 150, 160 };
	convnet_core::Tensor3D<double> tensor = utils::CreateTensorFromVec(vec);

	layer::ReLU relu(tensor, "ReLU forward");
	relu.Forward(relu.GetInput());
	std::cout << "Relu Forward:" << std::endl;
	convnet_core::PrintTensor(relu.GetOutput());

	std::cout << "Pool Forward:" << std::endl;
	layer::MaxPool pool(relu.GetOutput(), "pool", 2, 2);
	pool.Forward(relu.GetOutput());
	convnet_core::PrintTensor(pool.GetOutput());
	vec = std::vector<int>{ 1, 2, 3, 4 };
	convnet_core::Tensor3D<double> tensor_grad = utils::CreateTensorFromVec(vec, 2, 2);

	std::cout << "Pool Backprop:" << std::endl;
	pool.Backprop(tensor_grad);
	convnet_core::PrintTensor(pool.GetGrads());

	std::cout << "ReLU Backprop:" << std::endl;
	relu.Backprop(pool.GetGrads());
	convnet_core::PrintTensor(relu.GetGrads());

	return true;
}