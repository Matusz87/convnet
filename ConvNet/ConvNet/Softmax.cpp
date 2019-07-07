// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#include "Softmax.h"
#include <math.h>

namespace layer {
	// Default constuctor and destructor.
	Softmax::Softmax() { }
	Softmax::~Softmax() { }

	// Creates a Softmax layer invoking the base constructor.
	// @param name:		name of the layer
	// @param height:	input height
	// @param width:	input width
	// @param depth:	input depth
	Softmax::Softmax(std::string name, int height, int width, int depth)
				: Layer(name) {
		LayerType type = LayerType::Softmax;
		Layer::SetType(type);

		input = Tensor3D<double>(height, width, depth);
		output = Tensor3D<double>(height, width, depth);
		grad_input = Tensor3D<double>(height, width, depth);
	}
	
	// Copy constructor, used for loading parameters from a saved model.
	// param other: layer which will be copied.
	Softmax::Softmax(const Softmax& other) : Layer(other) { }

	// Creates a ReLU layer invoking the base constructor.
	// @param prev_act:	tensor from previous layer
	// @param name:		name of the layer
	Softmax::Softmax(convnet_core::Tensor3D<double>& prev_activation,
					 std::string name) : Layer(prev_activation, name) {
		convnet_core::Triplet shape = prev_activation.GetShape();
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
	}

	// Applies element-wise softmax function on previous activation map.
	// @param prev_act: activation map from previous layer
	void Softmax::Forward(const Tensor3D<double>& prev_activation) {
		/*assert(prev_activation.GetShape().width == 1 &&
			prev_activation.GetShape().depth == 1);*/
		input = Tensor3D<double>(prev_activation);
		double max = std::numeric_limits<double>::lowest();

		double sum_exp = 0;
		double exp_val = 0;
		
		for (int i = 0; i < input.GetShape().height; ++i) {
			exp_val = exp(input(i, 0, 0));
			output(i, 0, 0) = exp_val;
			sum_exp += exp_val;
		}

		output = output / sum_exp;
	}

	// Categorical Cross-entropy loss function.
	// @param target: one-hot encoded target tensor.
	double Softmax::Loss(Tensor3D<double>& target) {
		assert(target.GetShape().height == output.GetShape().height);

		for (int i = 0; i < output.GetShape().height; ++i) {
			if (target(i, 0, 0) == 1)
				return -log(output(i, 0, 0));
		}
		
	}

	// Calculates the gradients from the upstream gradient.
	// param grad_output: upstream gradient.
	void Softmax::Backprop(Tensor3D<double>& grad_out) {
		grad_input = grad_out;
	}

	// Not implemented, no trainable parameters.
	void Softmax::UpdateWeights(double learning_rate, double momentum) { }

	// Stores layer parameters in a JSON node.
	// returns layer: JSON representation of the layer. 
	nlohmann::json Softmax::Serialize() {
		nlohmann::json layer;

		layer["type"] = "softmax";
		layer["name"] = name;
		layer["height"] = GetInputShape().height;

		return layer;
	}
}
