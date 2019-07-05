// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#include "FC.h"

namespace layer {
	// Default constructor and destructor.
	FC::FC() { }
	FC::~FC() { }

	// Creates a FC layer invoking the base constructor.
	// @param name:			name of the layer
	// @param num_hidden:	number of neurons in the hidden layer.
	FC::FC(std::string name, int num_hidden) : Layer(name) {
		LayerType type = LayerType::FC;
		Layer::SetType(type);

		output = Tensor3D<double>(num_hidden, 1, 1);
		InitBias();
		has_weights_initialized = false;
	}

	// Creates a FC layer invoking the base constructor.
	// @param name:			name of the layer
	// @param num_input:	number of neurons in the input layer.
	// @param num_hidden:	number of neurons in the hidden layer.
	FC::FC(std::string name, int num_input, int num_hidden) : Layer(name) {
		LayerType type = LayerType::FC;
		Layer::SetType(type);

		input = Tensor3D<double>(num_input, 1, 1);
		this->num_hidden = num_hidden;
		output = Tensor3D<double>(num_hidden, 1, 1);
		InitWeights();
		InitBias();
		InitGrads();
		has_weights_initialized = true;
	}
	
	// Copy constructor, used for loading parameters from a saved model.
	// Hence, has_weight_initizalized is set to true.
	// param other: layer which will be copied.
	FC::FC(const FC& other) : Layer(other) { 
		this->num_hidden = other.num_hidden;
		weights = other.weights;
		bias = other.bias;
		InitGrads();

		has_weights_initialized = true;
	}

	// Creates a FC layer invoking the base constructor.
	// @param prev_activation:	tensor from previous layer
	// @param name:				layer name
	// @param num_hidden:		number of neurons in the hidden layer.
	FC::FC(convnet_core::Tensor3D<double>& prev_activation, std::string name,
		   int num_hidden) : Layer(prev_activation, name) {
		this->num_hidden = num_hidden;
		output = Tensor3D<double>(num_hidden, 1, 1);
		InitWeights();
		InitBias();
		InitGrads();
		has_weights_initialized = true;
	}

	// Forward pass. Performs the following operation: H(X) = Wx + b.
	// @param prev_act:	activation map from previous layer
	void FC::Forward(const Tensor3D<double>& prev_activation) 	{
		input = Tensor3D<double>(prev_activation);
		bool has_flattened = false;

		if (input.GetShape().width != 1 && input.GetShape().depth != 1) {
			tmp_input = input;
			input = input.Flatten();
			has_flattened = true;
		}
			
		if (!has_weights_initialized) {
			InitWeights();
			InitGrads();
		}
		
		for (int n = 0; n < output.GetShape().height; n++) {	
			double dot = 0;

			for (int i = 0; i < input.GetShape().height; i++)
				for (int j = 0; j < input.GetShape().width; j++) 
					for (int k = 0; k < input.GetShape().depth; k++) {
						int weight_index = MapToUnrolledIndex( i, j, k );
						dot += (input(i, j, k) * weights(weight_index, n, 0));
					}
			
			output(n, 0, 0) = dot + bias(n, 0, 0);;
		}

		// Input and its shape will be needed for backprop.
		if (has_flattened)
			input = tmp_input;
	}

	// Calculates the gradients from the upstream gradient.
	// Performs the following operations.
	// dX = dOut*W
	// dW = X*dOut
	// db = sum(dOut)
	// param grad_output: upstream gradient.
	void FC::Backprop(Tensor3D<double>& grad_output) {
		grad_input.InitZeros();
		// Handle flattened input.
		tmp_input = input;
		input = input.Flatten();

//std::cout << input.GetShape().height << " " << input.GetShape().width << " " << input.GetShape().depth << std::endl;
		double sum = 0;
		for (int n = 0; n < output.GetShape().height; n++) 		{
			for (int i = 0; i < input.GetShape().height; i++)
				for (int j = 0; j < input.GetShape().width; j++)
					for (int k = 0; k < input.GetShape().depth; k++) {
//std::cout << n << " " << i << " " << j << " " << k << std::endl;
						int weight_index = MapToUnrolledIndex(i, j, k);
						grad_input(i, j, k) += grad_output(n, 0, 0) * weights(weight_index, n, 0);
						grad_weights(weight_index, n, 0) = input(i, j, k) * grad_output(n, 0, 0);
					}
			sum += grad_output(n, 0, 0);
		}
		grad_bias = grad_bias + sum;

		// Restore input to its original shape
		input = tmp_input;
	}

	// Adjudsts weights based on the calculated gradients. 
	// Uses Nesterov Accelerated Gradient method.
	// @param lr:		learning rate
	// @param momentum: momentum
	// Further description of method: http://cs231n.github.io/neural-networks-3/#sgd
	void FC::UpdateWeights(double lr, double momentum) {
		Tensor3D<double> v_prev(velocities);
		velocities = velocities*momentum - (grad_weights*lr);
		weights = weights - (v_prev*momentum) + v_prev*(1 + momentum);
		
		bias = bias - (grad_bias*lr);
	}

	// Stores layer parameters in a JSON node.
	// returns layer: JSON representation of the layer. 
	nlohmann::json FC::Serialize() {
		nlohmann::json layer;
		nlohmann::json weights_json;
		nlohmann::json bias_json;

		layer["type"] = "fc";
		layer["name"] = name;
		// Take flattening into account.
		int inp_num = GetInputShape().height * GetInputShape().width * GetInputShape().depth;
		layer["input"] = inp_num;
		layer["output"] = GetOutputShape().height;

		for (int inp = 0; inp < inp_num; ++inp) {
			nlohmann::json weight;
			for (int out = 0; out < GetOutputShape().height; ++out)
				weight.push_back(weights(inp, out, 0));

			weights_json[std::to_string(inp)] = weight;
		}
		for (int out = 0; out < GetOutputShape().height; ++out)
			bias_json.push_back(bias(out, 0, 0));

		layer["weights"] = weights_json;
		layer["bias"] = bias_json;

		return layer;
	}

	// Not needed.
	double FC::Loss(Tensor3D<double>& target) { return 0.0; }

	// Getter methods.
	Tensor3D<double>& FC::GetWeights() 	{
		return weights;
	}

	Tensor3D<double>& FC::GetBias() {
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
	
	// He initialization.
	void FC::InitWeights() {
		convnet_core::Triplet input_shape = input.GetShape();
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
		velocities = Tensor3D<double>(inp_num, num_hidden, 1);
		velocities.InitZeros();
		
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