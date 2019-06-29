#include "FC.h"

namespace layer {
	FC::FC() { }
	FC::~FC() { }

	FC::FC(std::string name, int num_hidden) : Layer(name) {
		output = Tensor3D<double>(num_hidden, 1, 1);
		InitBias();
	}

	FC::FC(std::string name, int num_input, int num_hidden) : Layer(name) {
		input = Tensor3D<double>(num_input, 1, 1);
		this->num_hidden = num_hidden;
		output = Tensor3D<double>(num_hidden, 1, 1);
		InitWeights();
		InitBias();
		InitGrads();
	}

	FC::FC(convnet_core::Tensor3D<double>& prev_activation, std::string name,
		   int num_hidden) : Layer(prev_activation, name) {
		this->num_hidden = num_hidden;
		output = Tensor3D<double>(num_hidden, 1, 1);
		InitWeights();
		InitBias();
		InitGrads();
	}

	void FC::Forward(Tensor3D<double> prev_activation) 	{
		input = Tensor3D<double>(prev_activation);

		for (int n = 0; n < output.GetShape().height; n++) {	
			double dot = 0;

			for (int i = 0; i < input.GetShape().height; i++)
				for (int j = 0; j < input.GetShape().width; j++) 
					for (int k = 0; k < input.GetShape().depth; k++) {
						int weight_index = MapToUnrolledIndex( i, j, k );
						dot += (input(i, j, k) * weights(weight_index, n, 0));
					}
			
			output(n, 0, 0) = dot + bias(n, 0, 0);;
//			output_values[n] = dot + bias(n, 0, 0);;
		}
	}

	void FC::Backprop(Tensor3D<double> grad_output) {
		grad_input.InitZeros();
		// dX = dOut*W
		// dW = X*dOut
		// db = sum(dOut)

		double sum = 0;
		for (int n = 0; n < output.GetShape().height; n++) 		{
			for (int i = 0; i < input.GetShape().height; i++)
				for (int j = 0; j < input.GetShape().width; j++)
					for (int k = 0; k < input.GetShape().depth; k++) {
						int weight_index = MapToUnrolledIndex(i, j, k);
						grad_input(i, j, k) += grad_output(n, 0, 0) * weights(weight_index, n, 0);
						grad_weights(weight_index, n, 0) = input(i, j, k) * grad_output(n, 0, 0);
					}
			sum += grad_output(n, 0, 0);
		}
		grad_bias = grad_bias + sum;
	}

	Tensor3D<double>& FC::GetWeights() 	{
		return weights;
	}

	Tensor3D<double>& FC::GetBias()
	{
		return bias;
	}

	Tensor3D<double>& FC::GetGradWeights() 	{
		return grad_weights;
	}

	Tensor3D<double>& FC::GetGradBias() {
		return grad_bias;
	}

	Tensor3D<double> FC::GetGradInput() {
		return grad_input;
	}

	void FC::InitWeights() {
		convnet_core::Triplet input_shape = input.GetShape();
		std::cout << input_shape.height << " " << input_shape.width << " " << input_shape.depth << std::endl;
		weights = Tensor3D<double>(input_shape.height*input_shape.width*input_shape.depth,
						   num_hidden, 1);

		weights.InitRandom();
	}

	void FC::InitBias() {
			bias = Tensor3D<double>(num_hidden, 1, 1);
			bias.InitZeros();
	}

	void FC::InitGrads() {
		grad_input = Tensor3D<double>(input.GetShape().height,
									  input.GetShape().width, 
									  input.GetShape().depth);
		grad_input.InitZeros();
		int inp_num = input.GetShape().height*input.GetShape().width*input.GetShape().depth;
		grad_weights = Tensor3D<double>(inp_num, num_hidden, 1);
		grad_weights.InitZeros();

		grad_bias = Tensor3D<double>(num_hidden, 1, 1);
		grad_bias.InitZeros();
	}

	// Map original index of tensor to the unrolled version.
	int FC::MapToUnrolledIndex(int row, int col, int depth) {
		return depth * (input.GetShape().height * input.GetShape().width) +
			col * (input.GetShape().height) +
			row;
	}
}