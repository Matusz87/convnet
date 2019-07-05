#include "Layer.h"
#include <iostream>

namespace layer {
	Layer::Layer() {  }
	Layer::Layer(std::string name) {
		this->name = name; 
	}

	Layer::Layer(const convnet_core::Triplet& shape, std::string name) {
		this->name = name;
		input = Tensor3D<double>(shape.height, shape.width, shape.depth);
	}

	Layer::Layer(convnet_core::Tensor3D<double>& prev_activation, std::string name) {
		this->name = name;
		input = Tensor3D<double>(prev_activation);
	}

	Layer::Layer(const Layer& other) {
		input = other.input;
		output = other.output;
		grad_input = other.grad_input;
		name = other.name;
		type = other.type;
	}

	Layer::~Layer() { }
	convnet_core::Tensor3D<double>& Layer::GetInput() {
		return input;
	}
	convnet_core::Tensor3D<double>& Layer::GetOutput() {
		return output;
	}
	convnet_core::Tensor3D<double>& Layer::GetGrads() {
		return grad_input;
	}
	convnet_core::Triplet Layer::GetInputShape() {
		return input.GetShape();
	}
	convnet_core::Triplet Layer::GetOutputShape() {
		return output.GetShape();
	}
	convnet_core::Triplet Layer::GetGradsShape() {
		return grad_input.GetShape();
	}
	std::string Layer::GetName() {
		return name;
	}
	void Layer::SetType(LayerType type) {
		this->type = type;
	}

	LayerType Layer::GetType() {
		return type;
	}
}
