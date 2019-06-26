#include "TestReLU.h"
#include "ReLU.h"
#include "Utils.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

bool TestReLU::TestConstructor() {
	std::cout << "TestRelu::TestConstructor" << std::endl;
	convnet_core::Triplet shape;
	shape.height = 2;
	shape.width = 2;
	shape.depth = 1;
	
	layer::ReLU relu(shape, "ReLU 1");
	utils::PrintLayerShapes(relu);

	assert(relu.GetInputShape().width == shape.width && relu.GetInputShape().height == shape.height &&
		   relu.GetInputShape().depth == shape.depth &&
		   relu.GetOutputShape().width == shape.width && relu.GetOutputShape().height == shape.height &&
		   relu.GetOutputShape().depth == shape.depth &&
		   relu.GetGradsShape().width == shape.width && relu.GetGradsShape().height == shape.height &&
		   relu.GetGradsShape().depth == shape.depth);

	std::cout << "ReLU constructor test: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestReLU::TestConstructorWithTensor()
{
	std::cout << "TestRelu::TestConstructorWithTensor" << std::endl;

	std::vector<int> vec{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160 };
	Tensor3D<double> tensor = utils::CreateTensorFromVec(vec);
	convnet_core::Triplet shape = tensor.GetShape();

	layer::ReLU relu(tensor, "ReLUTensor");
	convnet_core::PrintTensor(relu.GetInput());
	utils::PrintLayerShapes(relu);

	assert(relu.GetInputShape().width == shape.width && relu.GetInputShape().height == shape.height &&
		relu.GetInputShape().depth == shape.depth &&
		relu.GetOutputShape().width == shape.width && relu.GetOutputShape().height == shape.height &&
		relu.GetOutputShape().depth == shape.depth &&
		relu.GetGradsShape().width == shape.width && relu.GetGradsShape().height == shape.height &&
		relu.GetGradsShape().depth == shape.depth);

	std::cout << "ReLU constructor test with tensor: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestReLU::TestConstructorWithMat() {
	std::cout << "TestRelu::TestConstructorWithMat" << std::endl;
	
	convnet_core::Tensor3D<double> tensor = utils::CreateTensorFromImage("../../../datasets/traffic_signs/1_0000_.bmp");
	convnet_core::Triplet shape = tensor.GetShape();

	layer::ReLU relu(tensor, "ReLUMat");
	utils::PrintLayerShapes(relu);

	assert(relu.GetInputShape().width == shape.width && relu.GetInputShape().height == shape.height &&
		relu.GetInputShape().depth == shape.depth &&
		relu.GetOutputShape().width == shape.width && relu.GetOutputShape().height == shape.height &&
		relu.GetOutputShape().depth == shape.depth &&
		relu.GetGradsShape().width == shape.width && relu.GetGradsShape().height == shape.height &&
		relu.GetGradsShape().depth == shape.depth);

	std::cout << "ReLU constructor test with Mat: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestReLU::TestForward() {
	std::cout << "TestRelu::TestForward" << std::endl;

	std::vector<int> vec{ 10, -20, 30, 40, -50, 60, 70, -80, 90, 100, -110, 120, 130, -140, 150, 160 };
	Tensor3D<double> tensor = utils::CreateTensorFromVec(vec);
	convnet_core::Triplet shape = tensor.GetShape();

	layer::ReLU relu(tensor, "ReLU forward");
	convnet_core::PrintTensor(relu.GetInput());
	utils::PrintLayerShapes(relu);

	relu.Forward(tensor);
	std::cout << "Ouptut afte forward propagation: ";
	convnet_core::PrintTensor(relu.GetOutput());

	std::cout << "ReLU forward test: SUCCESS" << std::endl << std::endl;

	return true;
}
