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
		Layer(std::string name);
		Layer(convnet_core::Triplet shape, std::string name);
		Layer(convnet_core::Tensor3D<double>& prev_activation, std::string name);
		~Layer();

		convnet_core::Tensor3D<double> GetInput();
		convnet_core::Tensor3D<double> GetOutput();
		convnet_core::Tensor3D<double> GetGrads();
		convnet_core::Triplet GetInputShape();
		convnet_core::Triplet GetOutputShape();
		convnet_core::Triplet GetGradsShape();

		virtual void Forward(Tensor3D<double> prev_activation) = 0;
		virtual void Backprop(Tensor3D<double> grad_output) = 0;
		virtual void UpdateWeights(double learning_rate) = 0;

	protected:
		Tensor3D<double> input;
		Tensor3D<double> output;
		// Gradient with respect to the input of the layer. 
		Tensor3D<double> grad_input;
		std::string name;
	};
}