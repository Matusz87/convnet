#include "TestConv.h"
#include "Conv.h"
#include "Utils.h"

TestConv::TestConv() { }
TestConv::~TestConv() { }

bool TestConv::TestConstructor() {
	std::cout << "TestConv::TestConstructor" << std::endl;
	convnet_core::Triplet shape;
	shape.height = 5; shape.width = 5; shape.depth = 3;
	int f_count = 2; int f_size = 3; 
	int stride = 2; int padding = 1;
	layer::Conv conv(shape, "conv_1", f_count, 
					 f_size, stride, padding);
	utils::PrintLayerShapes(conv);

	layer::Conv conv_prev(conv.GetInput(), "conv_prev1", f_count,
		f_size, stride, padding);
	utils::PrintLayerShapes(conv_prev);

	// same padding
	stride = 1;
	padding = (f_size - 1) / 2;
	layer::Conv conv2(shape, "conv_2", f_count,
					 f_size, stride, padding);
	utils::PrintLayerShapes(conv2);

	layer::Conv conv_prev2(conv2.GetInput(), "conv_prev2", f_count,
		f_size, stride, padding);
	utils::PrintLayerShapes(conv_prev2);



	return true;
}

bool TestConv::TestPadding() {
	/*std::vector<int> vec({ 1,2,3,4,5,6,7,8,9 });
	Tensor3D<double> tensor = utils::CreateTensorFromVec(vec, 3, 3);
	tensor = Tensor3D<double>(2, 2, 2);
	tensor.InitRandom();
	
	int f_count = 2; int f_size = 3;
	int stride = 2; int padding = 2;
	layer::Conv conv(tensor, "conv_1", f_count,
		f_size, stride, padding);
	
	std::cout << "Input: " << std::endl;
	convnet_core::PrintTensor(conv.GetInput());
	
	std::cout << "Padded: " << std::endl;
	Tensor3D<double> padded = conv.ZeroPad(conv.GetInput());
	convnet_core::PrintTensor(padded);
	*/
	return true;
}

bool TestConv::TestForward() {
	std::cout << "TestConv::TestForward" << std::endl;

	std::vector<int> vec({ 1,0,1,0,1,1,0,1,0 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 3, 3);
	std::vector<int> vec2({ 1,0,0,1 });
	Tensor3D<double> filter = utils::CreateTensorFromVec(vec2, 2, 2);

	int f_count = 1; int f_size = 2;
	int stride = 1; int padding = 0;
	layer::Conv conv(input, "conv_1", f_count,
		f_size, stride, padding);
	conv.GetWeights()[0] = filter;

	std::cout << "Input: " << std::endl;
	convnet_core::PrintTensor(conv.GetInput());
	std::cout << "Filter: " << std::endl;
	convnet_core::PrintTensor(conv.GetWeights()[0]);
	
	conv.Forward(conv.GetInput());
	std::cout << "Convolved: " << std::endl;
	convnet_core::PrintTensor(conv.GetOutput());

	return true;
}

bool TestConv::TestForwardPadded() {
	std::cout << "TestConv::TestForwardPadded" << std::endl;

	std::vector<int> vec({ 1,1,1,1 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 2, 2);
	std::vector<int> vec2({ 1,0,0,1 });
	Tensor3D<double> filter = utils::CreateTensorFromVec(vec2, 2, 2);

	int f_count = 1; int f_size = 2;
	int stride = 2; int padding = 1;
	layer::Conv conv(input, "conv_pad", f_count,
		f_size, stride, padding);
	conv.GetWeights()[0] = filter;

	std::cout << "Input: " << std::endl;
	convnet_core::PrintTensor(conv.GetInput());
	std::cout << "Filter: " << std::endl;
	convnet_core::PrintTensor(conv.GetWeights()[0]);

	conv.Forward(conv.GetInput());
	std::cout << "Convolved: " << std::endl;
	convnet_core::PrintTensor(conv.GetOutput());

	return true;
}

bool TestConv::TestForward2() {
	std::cout << std::endl << "TestConv::TestForward2" << std::endl;
	std::vector<int> vec({ 1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 4, 4);
	std::vector<int> vec2({ 1,0,0,1 });
	Tensor3D<double> filter = utils::CreateTensorFromVec(vec2, 2, 2);

	int f_count = 1; int f_size = 2;
	int stride = 1; int padding = 0;
	layer::Conv conv(input, "conv_1", f_count,
		f_size, stride, padding);
	conv.GetWeights()[0] = filter;

	std::cout << "Input: " << std::endl;
	convnet_core::PrintTensor(conv.GetInput());
	std::cout << "Filter: " << std::endl;
	convnet_core::PrintTensor(conv.GetWeights()[0]);

	conv.Forward(conv.GetInput());
	std::cout << "Convolved: " << std::endl;
	convnet_core::PrintTensor(conv.GetOutput());

	return true;
}

bool TestConv::TestForwardDeep() {
	std::cout << std::endl << "TestConv::TestForwardDeep" << std::endl;
	std::vector<int> vec({ 1,0,1,0,1,1,0,1,0 });
	std::vector<int> vec2({ 1,0,1,0,1,1,0,1,0 });
	Tensor3D<double> inp = utils::CreateTensorFromVec(vec, 3, 3);
	Tensor3D<double> inp2 = utils::CreateTensorFromVec(vec2, 3, 3);
	
	std::vector<Tensor3D<double>> tensor3D({ inp, inp2 });
	Tensor3D<double> input = utils::CreateTensorFrom3DVec(tensor3D, 3, 3);
	convnet_core::PrintTensor(input);


	std::vector<int> filt_vec1({ 1,0,0,1 });
	std::vector<int> filt_vec2({ 1,0,0,1 });
	Tensor3D<double> filter = utils::CreateTensorFromVec(filt_vec1, 2, 2);
	Tensor3D<double> filter2 = utils::CreateTensorFromVec(filt_vec2, 2, 2);

	//convnet_core::PrintTensor(filter);
	//convnet_core::PrintTensor(filter2);
	//convnet_core::PrintTensor(filter3);

	std::vector<Tensor3D<double>> filter3D({ filter, filter2 });

	//std::cout << "after tensor3D array: " << std::endl;
	Tensor3D<double> filters = utils::CreateTensorFrom3DVec(filter3D, 2, 2);
	//convnet_core::PrintTensor(filters);

	int f_count = 1; int f_size = 2;
	int stride = 1; int padding = 0;
	layer::Conv conv(input, "conv_deep", f_count,
		f_size, stride, padding);
	utils::PrintLayerShapes(conv);

	conv.GetWeights()[0] = filters;

	std::cout << "Input: " << std::endl;
	convnet_core::PrintTensor(conv.GetInput());
	std::cout << "Filters: " << std::endl;
	convnet_core::PrintTensor(conv.GetWeights()[0]);

	conv.Forward(conv.GetInput());
	std::cout << "Convolved: " << std::endl;
	convnet_core::PrintTensor(conv.GetOutput());

	return true;
}

// Example from Stanford cs231 (http://cs231n.github.io/convolutional-networks/#pool)
bool TestConv::TestForward3() {
	std::cout << std::endl << "TestConv::TestForward3" << std::endl;

	// Input tensor.
	std::vector<int> vec( { 1,0,1,1,0,2,1,1,0,2,2,2,2,0,1,2,2,1,2,0,0,1,2,2,1 });
	std::vector<int> vec2({ 1,1,1,2,1,0,0,1,1,1,2,0,1,2,0,1,2,2,1,2,2,1,2,0,1 });
	std::vector<int> vec3({ 0,2,0,1,0,2,2,0,2,0,1,1,0,0,1,2,0,2,2,2,2,2,2,0,0 });
	Tensor3D<double> inp = utils::CreateTensorFromVec(vec, 5, 5);
	Tensor3D<double> inp2 = utils::CreateTensorFromVec(vec2, 5, 5);
	Tensor3D<double> inp3 = utils::CreateTensorFromVec(vec3, 5, 5);
	
	std::vector<Tensor3D<double>> tensor3D({ inp, inp2, inp3 });
	Tensor3D<double> input = utils::CreateTensorFrom3DVec(tensor3D, 5, 5);
	convnet_core::PrintTensor(input);

	// First filter.
	std::vector<int> filt_vec1({ 0,-1,1,1,1,-1,0,1,1 });
	std::vector<int> filt_vec2({ 1,1,0,0,-1,0,0,0,0 });
	std::vector<int> filt_vec3({ 0,0,-1,1,1,0,1,1,-1 });
	Tensor3D<double> filter = utils::CreateTensorFromVec(filt_vec1, 3, 3);
	Tensor3D<double> filter2 = utils::CreateTensorFromVec(filt_vec2, 3, 3);
	Tensor3D<double> filter3 = utils::CreateTensorFromVec(filt_vec3, 3, 3);

	// Second filter.
	std::vector<int> filt_vec1_2({ 1,1,0,0,-1,-1,-1,0,1 });
	std::vector<int> filt_vec2_2({ 1,1,0,0,1,-1,0,0,1 });
	std::vector<int> filt_vec3_2({ -1,0,1,1,-1,-1,-1,-1,0 });
	Tensor3D<double> filter_2 = utils::CreateTensorFromVec(filt_vec1_2, 3, 3);
	Tensor3D<double> filter2_2 = utils::CreateTensorFromVec(filt_vec2_2, 3, 3);
	Tensor3D<double> filter3_2 = utils::CreateTensorFromVec(filt_vec3_2, 3, 3);

	std::vector<Tensor3D<double>> filter3D({ filter, filter2, filter3 });
	std::vector<Tensor3D<double>> filter3D_2({ filter_2, filter2_2, filter3_2 });

	Tensor3D<double> filters = utils::CreateTensorFrom3DVec(filter3D, 3, 3);
	Tensor3D<double> filters_2 = utils::CreateTensorFrom3DVec(filter3D_2, 3, 3);
	
	// First Bias (second is zero by default).
	Tensor3D<double> bias = Tensor3D<double>(1, 1, 1);
	bias(0, 0, 0) = 1;

	// Define the Conv layer.
	int f_count = 2; int f_size = 3;
	int stride = 2; int padding = 1;
	layer::Conv conv(input, "conv_1", f_count,
		f_size, stride, padding);
	utils::PrintLayerShapes(conv);

	conv.GetWeights()[0] = filters;
	conv.GetWeights()[1] = filters_2;
	conv.GetBias()[0] = bias;

	std::cout << "Input: " << std::endl;
	convnet_core::PrintTensor(conv.GetInput());
	std::cout << "Filters: " << std::endl;
	convnet_core::PrintTensor(conv.GetWeights()[0]);

	conv.Forward(conv.GetInput());
	std::cout << "Convolved: " << std::endl;
	convnet_core::PrintTensor(conv.GetOutput());

	return true;
}