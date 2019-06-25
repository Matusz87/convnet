#pragma once
#include <iostream>
#include "tensor3D.h"

using ::convnet_core::Tensor3D;

namespace layer {
	// Base class of the layers. 
	class Layer
	{
	public:
		Layer();
		Layer(convnet_core::Shape shape, std::string name);
		Layer(convnet_core::Tensor3D<double>& prevAct, std::string name);
		~Layer();

		//TODO: check const
		convnet_core::Tensor3D<double> GetInput();
		convnet_core::Tensor3D<double> GetOutput();
		convnet_core::Tensor3D<double> GetGrads();
		convnet_core::Shape GetInputShape();
		convnet_core::Shape GetOutputShape();
		convnet_core::Shape GetGradsShape();

		virtual void Forward(Tensor3D<double> prevActivation) = 0;
		virtual void Backprop(Tensor3D<double> dOutput) = 0;

	protected:
		Tensor3D<double> input;
		Tensor3D<double> output;
		// Gradients with respect to input tensor. 
		Tensor3D<double> dInput;
		std::string name;
	};
}