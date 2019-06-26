#include "TestMaxPool.h"
#include "MaxPool.h"
#include "Utils.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

bool TestMaxPool::TestConstructor() {
	std::cout << "TestMaxPool::TestConstructor" << std::endl;
	convnet_core::Triplet shape;
	shape.height = 4;
	shape.width = 4;
	shape.depth = 1;
	
	int stride = 2;
	int pool_size = 2;
	layer::MaxPool pool(shape, "Pool 1", stride, pool_size);
	int out_height = (shape.height - pool_size) / stride + 1;
	int out_width = (shape.width - pool_size) / stride + 1;

	utils::PrintLayerShapes(pool);

	assert(pool.GetInputShape().width == shape.width && pool.GetInputShape().height == shape.height &&
		pool.GetInputShape().depth == shape.depth &&
		pool.GetOutputShape().width == out_width && pool.GetOutputShape().height == out_height &&
		pool.GetOutputShape().depth == shape.depth &&
		pool.GetGradsShape().width == shape.width && pool.GetGradsShape().height == shape.height &&
		pool.GetGradsShape().depth == shape.depth);

	std::cout << "MaxPool constructor test: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestMaxPool::TestConstructorWithTensor() {
	std::cout << "TestMaxPool::TestConstructorWithTensor" << std::endl;
	std::vector<int> vec{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160 };
	
	convnet_core::Tensor3D<double> tensor = utils::CreateTensorFromVec(vec);
	convnet_core::Triplet shape = tensor.GetShape();
	int stride = 2;
	int pool_size = 2;
	int out_height = (shape.height - pool_size) / stride + 1;
	int out_width = (shape.width - pool_size) / stride + 1;

	layer::MaxPool pool(tensor, "Pool Tens", stride, pool_size);
	convnet_core::PrintTensor(pool.GetInput());
	utils::PrintLayerShapes(pool);

	assert(pool.GetInputShape().width == shape.width && pool.GetInputShape().height == shape.height &&
		pool.GetInputShape().depth == shape.depth &&
		pool.GetOutputShape().width == out_width && pool.GetOutputShape().height == out_height &&
		pool.GetOutputShape().depth == shape.depth &&
		pool.GetGradsShape().width == shape.width && pool.GetGradsShape().height == shape.height &&
		pool.GetGradsShape().depth == shape.depth);

	std::cout << "MaxPool constructor test with tensor: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestMaxPool::TestConstructorWithMat() {
	std::cout << "TestMaxPool::TestConstructorWithMat" << std::endl;
	
	convnet_core::Tensor3D<double> tensor = utils::CreateTensorFromImage("../../../datasets/traffic_signs/1_0000.bmp");
	convnet_core::Triplet shape = tensor.GetShape();
	
	int stride = 2;
	int pool_size = 2;
	int out_height = (shape.height - pool_size) / stride + 1;
	int out_width = (shape.width - pool_size) / stride + 1;

	layer::MaxPool pool(tensor, "Pool Mat", stride, pool_size);
	utils::PrintLayerShapes(pool);
	
	assert(pool.GetInputShape().width == shape.width && pool.GetInputShape().height == shape.height &&
		pool.GetInputShape().depth == shape.depth &&
		pool.GetOutputShape().width == out_width && pool.GetOutputShape().height == out_height &&
		pool.GetOutputShape().depth == shape.depth &&
		pool.GetGradsShape().width == shape.width && pool.GetGradsShape().height == shape.height &&
		pool.GetGradsShape().depth == shape.depth);

	std::cout << "MaxPool constructor test with Mat: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestMaxPool::TestForward() {
	std::cout << "TestMaxPool::Forward" << std::endl;
	
	std::vector<int> vec{ 10, -20, 30, 40, -50, 60, 70, -80, 90, 100, -110, 120, 130, -140, 150, 160 };
	convnet_core::Tensor3D<double> tensor = utils::CreateTensorFromVec(vec);
	convnet_core::Triplet shape = tensor.GetShape();
	int stride = 2;
	int pool_size = 2;
	int out_height = (shape.height - pool_size) / stride + 1;
	int out_width = (shape.width - pool_size) / stride + 1;

	layer::MaxPool pool(tensor, "Pool Forward", stride, pool_size);
	convnet_core::PrintTensor(pool.GetInput());
	utils::PrintLayerShapes(pool);

	pool.Forward(tensor);
	std::cout << "Output after forward propagation: ";
	convnet_core::PrintTensor(pool.GetOutput());

	std::cout << "MaxPool forward test: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestMaxPool::TestForwardWithMat() {
	std::cout << "TestMaxPool::TestForwardWithMat" << std::endl;

	convnet_core::Tensor3D<double> tensor = utils::CreateTensorFromImage("../../../datasets/traffic_signs/8.bmp");
	convnet_core::Triplet shape = tensor.GetShape();

	int stride = 4;
	int pool_size = 4;
	int out_height = (shape.height - pool_size) / stride + 1;
	int out_width = (shape.width - pool_size) / stride + 1;

	layer::MaxPool pool(tensor, "Pool Mat", stride, pool_size);

	convnet_core::PrintTensor(pool.GetInput());
	utils::PrintLayerShapes(pool);
	
	pool.Forward(pool.GetInput());
	std::cout << "Output after forward propagation: ";
	convnet_core::PrintTensor(pool.GetOutput());

	pool.Backprop(pool.GetOutput());
	std::cout << "Grad_in after backward propagation: ";
	convnet_core::PrintTensor(pool.GetGrads());

	std::cout << "MaxPool forward test: SUCCESS" << std::endl << std::endl;

	return true;
}

bool TestMaxPool::TestBackprop() {
	std::cout << "TestMaxPool::TestBackprop" << std::endl;
	
	std::vector<int> vec{ 10, -20, 30, 40, -50, 60, 70, -80, 90, 100, -110, 120, 130, -140, 150, 160 };
	convnet_core::Tensor3D<double> tensor = utils::CreateTensorFromVec(vec);
	convnet_core::Triplet shape = tensor.GetShape();
	int stride = 2;
	int pool_size = 2;
	int out_height = (shape.height - pool_size) / stride + 1;
	int out_width = (shape.width - pool_size) / stride + 1;

	layer::MaxPool pool(tensor, "Pool Backprop", stride, pool_size);
	convnet_core::PrintTensor(pool.GetInput());
	utils::PrintLayerShapes(pool);

	pool.Forward(tensor);
	std::cout << "Output afte forward propagation: ";
	convnet_core::PrintTensor(pool.GetOutput());

	vec = std::vector<int>{ 1, 2, 3, 4 };
	convnet_core::Tensor3D<double> tensor_grad = utils::CreateTensorFromVec(vec, 2, 2);

	pool.Backprop(tensor_grad);
	std::cout << "Grad_in after backward propagation: ";
	convnet_core::PrintTensor(pool.GetGrads());

	return true;
}
