#include "TestNet.h"

#include <algorithm>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include "Utils.h"
#include "tensor3D.h"
#include "Layer.h"
#include "ReLU.h"
#include "MaxPool.h"
#include "Conv.h"
#include "FC.h"
#include "Softmax.h"

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

	double lr = 0.01;
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

bool TestNet::TrainSignCE() {
	std::cout << "TestNet::TrainSignCE" << std::endl;
	std::vector<Tensor3D<double>> X;
	
	Tensor3D<double> input = utils::CreateTensorFromImage("../../../datasets/traffic_signs/train-52x52/1/1_0000.bmp");
	X.push_back(input);
	
	// Gradient w.r.t error.
	Tensor3D<double> d_out(6, 1, 1);

	layer::Conv conv(input, "conv", 4, 3, 1, 0);
	utils::PrintLayerShapes(conv);
	layer::ReLU relu("relu", 50, 50, 4);
	utils::PrintLayerShapes(relu);
	layer::MaxPool pool("pool", 50, 50, 4, 2, 2);
	utils::PrintLayerShapes(pool);
	layer::FC fc("fc", 25*25*4, 128);
	utils::PrintLayerShapes(fc);
	layer::ReLU relu_2("relu_2", 128, 1, 1);
	utils::PrintLayerShapes(relu);
	layer::FC fc_2("fc_2", 128, 6);
	utils::PrintLayerShapes(fc_2);
	layer::Softmax softmax("softmax", 6, 1, 1);

	Tensor3D<double> dW, db, target;
	utils::Dataset trainingSet = utils::GetTrainingSet();

	double lr = 0.003;
	double cum_loss = 0;
	int correct = 0;
	for (int epoch = 1; epoch <= 50; ++epoch) {
		std::srand(unsigned(std::time(0)));
		std::random_shuffle(trainingSet.begin(), trainingSet.end());

		cum_loss = 0;
		correct = 0;
		for (int m = 0; m < trainingSet.size(); ++m) {
			//std::cout << "Training sample " << m << std::endl;
			input = trainingSet[m].first;
			target = trainingSet[m].second;
			
			conv.Forward(input);
			relu.Forward(conv.GetOutput());
			pool.Forward(relu.GetOutput());
			fc.Forward(pool.GetOutput().Flatten());
			relu_2.Forward(fc.GetOutput());
			fc_2.Forward(relu_2.GetOutput());
			softmax.Forward(fc_2.GetOutput());

			if (epoch % 5 == 0) {
				/*std::cout << "Forward: " << m << std::endl;
				convnet_core::PrintTensor(softmax.GetOutput());*/
				if (utils::ComparePrediction(softmax.GetOutput(), target))
					++correct;				
			}

			// Calculate grads w.r.t loss function.
			// Loss function is Mean Absolute Error.
			d_out.InitZeros();
			d_out = (softmax.GetOutput() - target);
//			d_out = (fc.GetOutput() - target);

			// Derivative of loss function.
//			d_out = d_out.Sign();

			cum_loss += softmax.Loss(target);

			softmax.Backprop(d_out);
			fc_2.Backprop(softmax.GetGrads());
			relu_2.Backprop(fc_2.GetGrads());
			fc.Backprop(relu_2.GetGrads());
//			fc.Backprop(d_out);
			pool.Backprop(fc.GetGrads().Reshape(pool.GetOutputShape()));
			relu.Backprop(pool.GetGrads());
			conv.Backprop(relu.GetGrads());

			// Update weights.
			fc_2.UpdateWeights(lr);
			fc.UpdateWeights(lr);

			// Conv layer update
			conv.UpdateWeights(lr);
		}
		
		std::cout << std::endl;
		std::cout << "Epoch " << epoch << " is done. " << std::endl;
		std::cout << "Loss: " << cum_loss / (double)X.size() << std::endl << std::endl;
		if (epoch %5 == 0)
			std::cout << "Training accuracy: " << correct / (double)trainingSet.size() << std::endl;
	}

	std::cout << std::endl << "FC2 output" << std::endl;
	convnet_core::PrintTensor(fc_2.GetOutput());

	return true;
}

bool TestNet::TrainSignCE2() {
	std::cout << "TestNet::TrainSignCE2x2" << std::endl;
	std::vector<Tensor3D<double>> X;

	Tensor3D<double> input = utils::CreateTensorFromImage("../../../datasets/traffic_signs/train-52x52/1/1_0000.bmp");
	X.push_back(input);

	// Gradient w.r.t error.
	Tensor3D<double> d_out(6, 1, 1);

	layer::MaxPool pool_0(input, "pool_0", 2, 2);
	utils::PrintLayerShapes(pool_0);
	layer::Conv conv(26, 26, 3, "conv", 12, 3, 1, 0);
	utils::PrintLayerShapes(conv);
	layer::ReLU relu("relu", 24, 24, 12);
	utils::PrintLayerShapes(relu);
	layer::MaxPool pool("pool", 24, 24, 12, 2, 2);
	utils::PrintLayerShapes(pool);
	layer::FC fc("fc", 12 * 12 * 12, 64);
	utils::PrintLayerShapes(fc);
	layer::ReLU relu_2("relu_2", 64, 1, 1);
	utils::PrintLayerShapes(relu_2);
	layer::FC fc_2("fc_2", 64, 12);
	utils::PrintLayerShapes(fc_2);
	layer::Softmax softmax("softmax", 12, 1, 1);

	Tensor3D<double> dW, db, target;
	utils::Dataset trainingSet = utils::GetTrainingSet();

	double lr = 0.0001;
	double cum_loss = 0;
	int correct = 0;
	for (int epoch = 1; epoch <= 50; ++epoch) {
		std::srand(unsigned(std::time(0)));
		std::random_shuffle(trainingSet.begin(), trainingSet.end());

		cum_loss = 0;
		correct = 0;
		for (int m = 0; m < trainingSet.size(); ++m) {
			//std::cout << "Training sample " << m << std::endl;
			input = trainingSet[m].first;
			target = trainingSet[m].second;

			pool_0.Forward(input);
			conv.Forward(pool_0.GetOutput());
			relu.Forward(conv.GetOutput());
			pool.Forward(relu.GetOutput());
			fc.Forward(pool.GetOutput().Flatten());
			relu_2.Forward(fc.GetOutput());
			fc_2.Forward(relu_2.GetOutput());
			softmax.Forward(fc_2.GetOutput());

			if (epoch % 5 == 0) {
				/*std::cout << "Forward: " << m << std::endl;
				convnet_core::PrintTensor(softmax.GetOutput());*/
				if (utils::ComparePrediction(softmax.GetOutput(), target))
					++correct;
			}

			// Calculate grads w.r.t loss function.
			// Loss function is Mean Absolute Error.
			d_out.InitZeros();
			d_out = (softmax.GetOutput() - target);
			//			d_out = (fc.GetOutput() - target);

			// Derivative of loss function.
			//			d_out = d_out.Sign();

			cum_loss += softmax.Loss(target);

			softmax.Backprop(d_out);
			fc_2.Backprop(softmax.GetGrads());
			relu_2.Backprop(fc_2.GetGrads());
			fc.Backprop(relu_2.GetGrads());
			//			fc.Backprop(d_out);
			pool.Backprop(fc.GetGrads().Reshape(pool.GetOutputShape()));
			relu.Backprop(pool.GetGrads());
			conv.Backprop(relu.GetGrads());

			// Update weights.
			fc_2.UpdateWeights(lr);
			fc.UpdateWeights(lr);

			// Conv layer update
			conv.UpdateWeights(lr);
		}

		std::cout << std::endl;
		std::cout << "Epoch " << epoch << " is done. " << std::endl;
		std::cout << "Loss: " << cum_loss / (double)X.size() << std::endl << std::endl;
		if (epoch % 5 == 0)
			std::cout << "Training accuracy: " << correct / (double)trainingSet.size() << std::endl;
	}

	std::cout << std::endl << "FC2 output" << std::endl;
	convnet_core::PrintTensor(fc_2.GetOutput());

	return true;
}


bool TestNet::TrainSign() {
	std::cout << "TestNet::TrainSign" << std::endl;

	convnet_core::Tensor3D<double> input = utils::CreateTensorFromImage("../../../datasets/traffic_signs/1_0000.bmp");

	std::vector<int> vec_t({ 1,0,0,0,0,0,0,0,0,0 });
	Tensor3D<double> target = utils::CreateTensorFromVec(vec_t, 10, 1);

	// Gradient w.r.t error.
	Tensor3D<double> d_out(10, 1, 1);

	layer::Conv conv(input, "conv", 32, 3, 1, 0);
	utils::PrintLayerShapes(conv);
	layer::ReLU relu("relu", 50, 50, 32);
	utils::PrintLayerShapes(relu);
	layer::MaxPool pool("pool", 50, 50, 32, 2, 2);
	utils::PrintLayerShapes(pool);
	layer::FC fc("fc", 25 * 25 * 32, 10);
	utils::PrintLayerShapes(fc);
	//	layer::ReLU relu-fc("relu-fc", 128, 1, 1);
	//	layer::FC fc2("fc2", 128, 10);

	Tensor3D<double> dW, db;

	double lr = 0.00001;
	for (int epoch = 0; epoch < 10; ++epoch) {
		conv.Forward(input);
		relu.Forward(conv.GetOutput());
		pool.Forward(relu.GetOutput());
		fc.Forward(pool.GetOutput().Flatten());

		std::cout << std::endl << "Forward: " << std::endl;
		convnet_core::PrintTensor(fc.GetOutput());

		// Calculate grads w.r.t loss function.
		// Loss function is Mean Absolute Error.
		d_out.InitZeros();
		d_out = (fc.GetOutput() - target);

		// Derivative of loss function.
		d_out = d_out.Sign();

		std::cout << "Loss: " << d_out.Sum() << std::endl;

		fc.Backprop(d_out);
		pool.Backprop(fc.GetGrads().Reshape(pool.GetOutputShape()));
		relu.Backprop(pool.GetGrads());
		conv.Backprop(relu.GetGrads());

		// Update weights.
		dW = fc.GetGradWeights();
		dW = dW * lr;
		fc.GetWeights() = fc.GetWeights() - dW;
		db = fc.GetGradBias();
		db = db * lr;
		fc.GetBias() = fc.GetBias() - db;

		// Conv layer update
		for (int i = 0; i < conv.GetGradWeights().size(); ++i) {
			dW = conv.GetGradWeights()[i];
			dW = dW * lr;
			conv.GetWeights()[i] = conv.GetWeights()[i] - dW;
		}

		for (int i = 0; i < conv.GetBias().size(); ++i) {
			db = conv.GetGradBias()[i];
			db = db * lr;
			conv.GetBias()[i] = conv.GetBias()[i] - db;
		}

		std::cout << "Epoch " << epoch << " is done. " << std::endl;
	}

	return true;
}