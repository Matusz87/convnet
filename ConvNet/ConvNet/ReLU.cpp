// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#include "ReLU.h"

namespace layer {
	// Default consturctor and destructor.
	ReLU::ReLU() { }
	ReLU::~ReLU() { }

	// Creates a ReLU layer invoking the base constructor.
	// @param height:	input height
	// @param width:	input width
	// @param depth:	input depth
	// @param name:		name of the layer
	ReLU::ReLU(std::string name, int height, int width, int depth) : Layer(name)	{
		LayerType type = LayerType::ReLU;
		Layer::SetType(type);

		input = Tensor3D<double>(height, width, depth);
		output = Tensor3D<double>(height, width, depth);
		grad_input = Tensor3D<double>(height, width, depth);
	}

	// Copy constructor, used for loading parameters from a saved model.
	// param other: layer which will be copied.
	ReLU::ReLU(const ReLU& other) : Layer(other) { }

	// Creates a ReLU layer invoking the base constructor.
	// @param shape:	input shape
	// @param name:		name of the layer
	ReLU::ReLU(convnet_core::Triplet shape, std::string name) : Layer(shape, name) {
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
	}

	// Creates a ReLU layer invoking the base constructor.
	// @param prev_act:	tensor from previous layer
	// @param name:		name of the layer
	ReLU::ReLU(convnet_core::Tensor3D<double>& prev_activation, std::string name) 
			: Layer(prev_activation, name) {
		convnet_core::Triplet shape = prev_activation.GetShape();
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
	}

	// Applies rectified linear unit non-linearity on previous activation map.
	// @param prev_act: activation map from previous layer
	void ReLU::Forward(const Tensor3D<double>& prev_activation) {
		input = Tensor3D<double>(prev_activation);
		
		int depth = input.GetShape().depth;
		int height = input.GetShape().height;
		int width = input.GetShape().width;
		output = Tensor3D<double>(height, width, depth);

		for (int k = 0; k < depth; ++k) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					//output(i, j, k) = input(i, j, k) > 0 ? input(i, j, k) : 0;
					output(i, j, k) = input(i, j, k) < 0 ? (0.1*input(i, j, k)) : (1 * input(i, j, k));

				}
			}
		}
	}

	// Calculates the gradients from the upstream gradient.
	// param grad_output: upstream gradient.
	void ReLU::Backprop(Tensor3D<double>& grad_out) {
		grad_input.InitZeros();

		int depth = input.GetShape().depth;
		int height = input.GetShape().height;
		int width = input.GetShape().width;
		for (int k = 0; k < depth; ++k) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					//grad_input(i, j, k) = input(i, j, k) < 0 ? 0 : (1 * grad_out(i, j, k));
					grad_input(i, j, k) = input(i, j, k) < 0 ? (0.1*grad_out(i, j, k)) : (1 * grad_out(i, j, k));
				}
			}
		}
	}

	// Not implemented, no trainable parameters.
	void ReLU::UpdateWeights(double lr, double momentum) { }

	// Stores layer parameters in a JSON node.
	// returns layer: JSON representation of the layer. 
	nlohmann::json ReLU::Serialize() {
		nlohmann::json layer;

		layer["type"] = "relu";
		layer["name"] = name;
		layer["height"] = GetInputShape().height;
		layer["width"] = GetInputShape().width;
		layer["depth"] = GetInputShape().depth;

		return layer;
	}

	// Not implemented.
	double ReLU::Loss(Tensor3D<double>& target) { return 0.0; }
}


