#include "Layer.h"
#include <iostream>

namespace layer {
	Layer::Layer() { }

	Layer::Layer(convnet_core::Shape shape, std::string name) {
		std::cout << "Layer shape constructor" << std::endl;
		this->name = name;
		std::cout << "Layer name: " << this->name << std::endl;

		input = Tensor3D<double>(shape.height, shape.width, shape.depth);
		std::cout << "Input shape: ";
		convnet_core::PrintShape(GetInputShape());
	}

	Layer::Layer(convnet_core::Tensor3D<double>& prevAct, std::string name) : input(prevAct) {
		std::cout << "Layer prevAct constructor" << std::endl;
		this->name = name;
		std::cout << "Layer name: " << this->name << std::endl;
		
		input = Tensor3D<double>(prevAct);

		std::cout << "Input shape: ";
		convnet_core::PrintShape(GetInputShape());
	}


	Layer::~Layer() { }
	convnet_core::Tensor3D<double> Layer::GetInput() {
		return input;
	}
	convnet_core::Tensor3D<double> Layer::GetOutput() {
		return output;
	}
	convnet_core::Tensor3D<double> Layer::GetGrads() {
		return dInput;
	}
	convnet_core::Shape Layer::GetInputShape() {
		return input.GetShape();
	}
	convnet_core::Shape Layer::GetOutputShape() {
		return output.GetShape();
	}
	convnet_core::Shape Layer::GetGradsShape() {
		return dInput.GetShape();
	}
}
