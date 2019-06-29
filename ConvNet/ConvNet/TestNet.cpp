#include "TestNet.h"

#include <iostream>
#include "Utils.h"
#include "tensor3D.h"
#include "Layer.h"
#include "ReLU.h"
#include "MaxPool.h"
#include "Conv.h"
#include "FC.h"

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

bool TestNet::TrainConvLayer() {
	std::cout << "TestConv::TestBackprop" << std::endl;

	std::vector<int> vec({ 1,0,1,0,1,1,0,1,0 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 3, 3);

	std::vector<int> inp_vec_2({ 1,0,0,1,1,0,1,0,1 });
	Tensor3D<double> input_2 = utils::CreateTensorFromVec(inp_vec_2, 3, 3);

	// Gradient w.r.t error.
	Tensor3D<double> d_out(2,2,1);

	// Create input tensor.
	std::vector<Tensor3D<double>> tensor3D({ input, input_2 });
	input = utils::CreateTensorFrom3DVec(tensor3D, 3, 3);

	//std::vector<int> vec_4({ 3,2,1,2 });
	std::vector<int> vec_4({ 1,-1,1,0 });
	Tensor3D<double> target = utils::CreateTensorFromVec(vec_4, 2, 2);

	int f_count = 1; int f_size = 2;
	int stride = 1; int padding = 0;
	layer::Conv conv(input, "conv_1", f_count, f_size, stride, padding);
	utils::PrintLayerShapes(conv);
	std::cout << conv.GetWeights()[0].GetShape().depth << std::endl;

	Tensor3D<double> dW, db;
	
	double lr = 0.015;
	for (int i = 0; i < 500; ++i) {
		conv.Forward(conv.GetInput());
		std::cout << "Convolved: " << std::endl;
		convnet_core::PrintTensor(conv.GetOutput());
		
		// Calculate grads w.r.t loss function.
		// Loss function is Mean Absolute Error.
		d_out.InitZeros();
		d_out = (conv.GetOutput() - target);
		// Derivative of loss function.
		d_out = d_out.Sign();

		std::cout << "Error: " << std::endl;
		convnet_core::PrintTensor(d_out);

		std::cout << "Loss: " << d_out.Sum() << std::endl;

		conv.Backprop(d_out);

		std::cout << "Grads w.r.t weight: " << std::endl;
		convnet_core::PrintTensor(conv.GetGradWeights()[0]);

		std::cout << "Grads w.r.t input: " << std::endl;
		convnet_core::PrintTensor(conv.GetGradInput());	

		// Update weights.
		for (int i = 0; i < conv.GetGradWeights().size(); ++i) {
			dW = conv.GetGradWeights()[i];
			dW = dW * lr;
			conv.GetWeights()[i] = conv.GetWeights()[i] - dW;
		}

		// TODO: update bias
		for (int i = 0; i < conv.GetBias().size(); ++i) {
			db = conv.GetGradBias()[i];
			db = db * lr;
			conv.GetBias()[i] = conv.GetBias()[i] - db;
		}

		std::cout << "Updated weight: " << std::endl;
		convnet_core::PrintTensor(conv.GetWeights()[0]);

		std::cout << "Updated bias: " << std::endl;
		convnet_core::PrintTensor(conv.GetBias()[0]);

		std::cout << "GetGradInput: " << std::endl;
		convnet_core::PrintTensor(conv.GetGradInput());

		std::cout << std::endl;
	}
	
	return false;
}

bool TestNet::TrainFC() {
	std::cout << "TestNet::TrainFC" << std::endl;

	std::vector<int> vec({ 1,2, 3 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 3, 1);

	std::vector<int> w_vec({ 1,2,3,4,5,6 });
	Tensor3D<double> weights = utils::CreateTensorFromVec(w_vec, 3, 2);

	std::vector<int> e_vec({ 3,2 });
	Tensor3D<double> error = utils::CreateTensorFromVec(e_vec, 2, 1);

	std::vector<int> vec_t({ 25, 30 });
	Tensor3D<double> target = utils::CreateTensorFromVec(vec_t, 2, 1);

	// Gradient w.r.t error.
	Tensor3D<double> d_out(2, 1, 1);

	//Tensor3D<double> tensor(2, 1, 1);
	layer::FC fc(input, "fc", 2);
	//fc.GetWeights() = weights;

	Tensor3D<double> dW, db;

	double lr = 0.015;
	for (int i = 0; i < 100; ++i) {
		fc.Forward(fc.GetInput());
		std::cout << "Forward: " << std::endl;
		convnet_core::PrintTensor(fc.GetOutput());

		// Calculate grads w.r.t loss function.
		// Loss function is Mean Absolute Error.
		d_out.InitZeros();
		d_out = (fc.GetOutput() - target);
		// Derivative of loss function.
		d_out = d_out.Sign();

		std::cout << "Error: " << std::endl;
		convnet_core::PrintTensor(d_out);

		std::cout << "Loss: " << d_out.Sum() << std::endl;

		fc.Backprop(d_out);

		std::cout << "Grads w.r.t weight: " << std::endl;
		convnet_core::PrintTensor(fc.GetGradWeights());

		std::cout << "Grads w.r.t input: " << std::endl;
		convnet_core::PrintTensor(fc.GetGradInput());

		// Update weights.
		dW = fc.GetGradWeights();
		dW = dW * lr;
		fc.GetWeights() = fc.GetWeights() - dW;

		// TODO: update bias
		db = fc.GetGradBias();
		db = db * lr;
		fc.GetBias() = fc.GetBias() - db;

		std::cout << "Updated weight: " << std::endl;
		convnet_core::PrintTensor(fc.GetWeights());

		std::cout << "Updated bias: " << std::endl;
		convnet_core::PrintTensor(fc.GetBias());

		std::cout << "GetGradInput: " << std::endl;
		convnet_core::PrintTensor(fc.GetGradInput());

		std::cout << std::endl;
	}

	return true;
}

bool TestNet::TrainFC2() {
	std::cout << "TestNet::TrainFC2" << std::endl;
	// FC(100) -> ReLU -> FC(2)

	std::vector<int> vec({ 1, 2, 3 });
	Tensor3D<double> input = utils::CreateTensorFromVec(vec, 3, 1);

	std::vector<int> w_vec({ 1,2,3,4,5,6 });
	Tensor3D<double> weights = utils::CreateTensorFromVec(w_vec, 3, 2);

	std::vector<int> e_vec({ 3,2 });
	Tensor3D<double> error = utils::CreateTensorFromVec(e_vec, 2, 1);

	std::vector<int> vec_t({ 25, 30 });
	Tensor3D<double> target = utils::CreateTensorFromVec(vec_t, 2, 1);

	// Gradient w.r.t error.
	Tensor3D<double> d_out(2, 1, 1);
	
	layer::FC fc(input, "fc", 10);
	layer::ReLU relu("relu", 10, 1, 1);
	layer::FC fc2("fc2", 10, 2);

	Tensor3D<double> dW, db;

	double lr = 0.02;
	for (int i = 0; i < 50; ++i) {
		fc.Forward(fc.GetInput());
//		std::cout << "FC shape: " << std::endl;
		utils::PrintLayerShapes(fc);
//		std::cout << "FC2 shape: " << std::endl;
		utils::PrintLayerShapes(fc2);

		relu.Forward(fc.GetOutput());
		fc2.Forward(relu.GetOutput());
		
		std::cout << "Forward: " << std::endl;
		convnet_core::PrintTensor(fc2.GetOutput());

		// Calculate grads w.r.t loss function.
		// Loss function is Mean Absolute Error.
		d_out.InitZeros();
		d_out = (fc2.GetOutput() - target);
		// Derivative of loss function.
		d_out = d_out.Sign();

		std::cout << "Error: " << std::endl;
		convnet_core::PrintTensor(d_out);

		std::cout << "Loss: " << d_out.Sum() << std::endl;

		fc2.Backprop(d_out);
		relu.Backprop(fc2.GetGradInput());
		fc.Backprop(relu.GetGrads());

		std::cout << "Grads w.r.t weight: " << std::endl;
		convnet_core::PrintTensor(fc.GetGradWeights());

		std::cout << "Grads w.r.t input: " << std::endl;
		convnet_core::PrintTensor(fc.GetGradInput());

		// Update weights.
		dW = fc2.GetGradWeights();
		dW = dW * lr;
		fc2.GetWeights() = fc2.GetWeights() - dW;
		db = fc2.GetGradBias();
		db = db * lr;
		fc2.GetBias() = fc2.GetBias() - db;

		dW = fc.GetGradWeights();
		dW = dW * lr;
		fc.GetWeights() = fc.GetWeights() - dW;
		db = fc.GetGradBias();
		db = db * lr;
		fc.GetBias() = fc.GetBias() - db;

		std::cout << "Updated weight: " << std::endl;
		convnet_core::PrintTensor(fc.GetWeights());

		//std::cout << "Updated bias: " << std::endl;
		//convnet_core::PrintTensor(fc.GetBias());

		std::cout << "GetGradInput: " << std::endl;
		convnet_core::PrintTensor(fc.GetGradInput());

		std::cout << std::endl;
	}

	return true;
}

bool TestNet::TrainMNISTdigit() {
	std::cout << "TestNet::TrainMNISTdigit" << std::endl;
	// FC(128) -> ReLU -> FC(10)

	convnet_core::Tensor3D<double> input = utils::CreateTensorFromImage("../../../datasets/5.png");

	std::vector<int> vec_t({ 0,0,0,0,0,1,0,0,0,0 });
	Tensor3D<double> target = utils::CreateTensorFromVec(vec_t, 10, 1);

	// Gradient w.r.t error.
	Tensor3D<double> d_out(10, 1, 1);

	layer::FC fc(input, "fc", 128);
	layer::ReLU relu("relu", 128, 1, 1);
	layer::FC fc2("fc2", 128, 10);

	Tensor3D<double> dW, db;

	double lr = 0.001;
	for (int i = 0; i < 50; ++i) {
		fc.Forward(fc.GetInput());
		utils::PrintLayerShapes(fc);
		utils::PrintLayerShapes(fc2);

		relu.Forward(fc.GetOutput());
		fc2.Forward(relu.GetOutput());

		std::cout << std::endl << "Forward: " << std::endl;
		convnet_core::PrintTensor(fc2.GetOutput());

		// Calculate grads w.r.t loss function.
		// Loss function is Mean Absolute Error.
		d_out.InitZeros();
		d_out = (fc2.GetOutput() - target);
		// Derivative of loss function.
		d_out = d_out.Sign();

		/*std::cout << "Error: " << std::endl;
		convnet_core::PrintTensor(d_out);*/

		std::cout << "Loss: " << d_out.Sum() << std::endl;

		fc2.Backprop(d_out);
		relu.Backprop(fc2.GetGradInput());
		fc.Backprop(relu.GetGrads());

		// Update weights.
		dW = fc2.GetGradWeights();
		dW = dW * lr;
		fc2.GetWeights() = fc2.GetWeights() - dW;
		db = fc2.GetGradBias();
		db = db * lr;
		fc2.GetBias() = fc2.GetBias() - db;

		dW = fc.GetGradWeights();
		dW = dW * lr;
		fc.GetWeights() = fc.GetWeights() - dW;
		db = fc.GetGradBias();
		db = db * lr;
		fc.GetBias() = fc.GetBias() - db;

		std::cout << std::endl;
	}

	return true;
}

bool TestNet::TrainSignFC() {
	std::cout << "TestNet::TrainSignFC" << std::endl;
	// FC(128) -> ReLU -> FC(10)

	convnet_core::Tensor3D<double> input = utils::CreateTensorFromImage("../../../datasets/traffic_signs/1_0000.bmp");

	std::vector<int> vec_t({ 1,0,0,0,0,0,0,0,0,0 });
	Tensor3D<double> target = utils::CreateTensorFromVec(vec_t, 10, 1);

	// Gradient w.r.t error.
	Tensor3D<double> d_out(10, 1, 1);

	layer::FC fc(input, "fc", 128);
	layer::ReLU relu("relu", 128, 1, 1);
	layer::FC fc2("fc2", 128, 10);

	Tensor3D<double> dW, db;

	double lr = 0.001;
	for (int i = 0; i < 50; ++i) {
		fc.Forward(fc.GetInput());
		/*utils::PrintLayerShapes(fc);
		utils::PrintLayerShapes(fc2);*/

		relu.Forward(fc.GetOutput());
		fc2.Forward(relu.GetOutput());

		std::cout << std::endl << "Forward: " << std::endl;
		convnet_core::PrintTensor(fc2.GetOutput());

		// Calculate grads w.r.t loss function.
		// Loss function is Mean Absolute Error.
		d_out.InitZeros();
		d_out = (fc2.GetOutput() - target);
		// Derivative of loss function.
		d_out = d_out.Sign();

		/*std::cout << "Error: " << std::endl;
		convnet_core::PrintTensor(d_out);*/

		std::cout << "Loss: " << d_out.Sum() << std::endl;

		fc2.Backprop(d_out);
		relu.Backprop(fc2.GetGradInput());
		fc.Backprop(relu.GetGrads());

		// Update weights.
		dW = fc2.GetGradWeights();
		dW = dW * lr;
		fc2.GetWeights() = fc2.GetWeights() - dW;
		db = fc2.GetGradBias();
		db = db * lr;
		fc2.GetBias() = fc2.GetBias() - db;

		dW = fc.GetGradWeights();
		dW = dW * lr;
		fc.GetWeights() = fc.GetWeights() - dW;
		db = fc.GetGradBias();
		db = db * lr;
		fc.GetBias() = fc.GetBias() - db;

		std::cout << std::endl;
	}

	return true;
}