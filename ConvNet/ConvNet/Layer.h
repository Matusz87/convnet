#pragma once
#include <iostream>
#include "tensor3D.h"
#include <nlohmann\json.hpp>

using ::convnet_core::Tensor3D;

namespace layer {
	// Base class of the layers. 
	enum class LayerType { Conv, ReLU, Pool, FC, Softmax };

	class Layer
	{
	public:
		Layer();
		Layer(std::string name);
		Layer(const convnet_core::Triplet& shape, std::string name);
		Layer(Tensor3D<double>& prev_activation, std::string name);
		Layer(const Layer& other);
		~Layer();

		Tensor3D<double>& GetInput();
		Tensor3D<double>& GetOutput();
		Tensor3D<double>& GetGrads();
		convnet_core::Triplet GetInputShape();
		convnet_core::Triplet GetOutputShape();
		convnet_core::Triplet GetGradsShape();
		std::string GetName();
		void SetType(LayerType type);
		LayerType GetType();

		virtual void Forward(const Tensor3D<double>& prev_activation) = 0;
		virtual void Backprop(Tensor3D<double>& grad_output) = 0;
		virtual void UpdateWeights(double learning_rate, double momentum = 0.9) = 0;
		virtual nlohmann::json Serialize() = 0;
		virtual double Loss(Tensor3D<double>& target) = 0;

	protected:
		Tensor3D<double> input;
		Tensor3D<double> output;
		// Gradient with respect to the input of the layer. 
		Tensor3D<double> grad_input;
		std::string name;
		LayerType type;
	};
}