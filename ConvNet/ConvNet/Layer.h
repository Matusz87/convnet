// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#pragma once
#include <iostream>
#include "tensor3D.h"
#include <nlohmann\json.hpp>

using ::convnet_core::Tensor3D;

namespace layer {
	// Enum for specific layer types.
	enum class LayerType { Conv, ReLU, Pool, FC, Softmax };

	// Base class of the layers. Specific layers will be inherited from this class.
	// Easily extensible with new layers, four virtual methods have to be overridden.
	class Layer
	{
	public:
		Layer();
		Layer(std::string name);
		Layer(const convnet_core::Triplet& shape, std::string name);
		Layer(Tensor3D<double>& prev_activation, std::string name);
		Layer(const Layer& other);
		~Layer();

		// Getter methods.
		Tensor3D<double>& GetInput();
		Tensor3D<double>& GetOutput();
		Tensor3D<double>& GetGrads();
		convnet_core::Triplet GetInputShape();
		convnet_core::Triplet GetOutputShape();
		convnet_core::Triplet GetGradsShape();
		std::string GetName();
		void SetType(LayerType type);
		LayerType GetType();

		// Core functionality of a layer. Specific layer subclasses
		// have to override these methods.

		// Forward propagation.
		virtual void Forward(const Tensor3D<double>& prev_activation) = 0;
		// Backpropagation for obtaining gradients.
		virtual void Backprop(Tensor3D<double>& grad_output) = 0;
		// Adjusts weigths based on the gradients obtained by backprop.
		virtual void UpdateWeights(double learning_rate, double momentum = 0.9) = 0;
		// Serialization method for saving layer parameters.
		virtual nlohmann::json Serialize() = 0;
		// Returns the loss with respect to the given loss function and target.
		virtual double Loss(Tensor3D<double>& target) = 0;

	protected:
		// Input tensor of a layer, either an image or the output of the previous layer.
		Tensor3D<double> input;
		// Output tensor that stores the result of the forward pass.
		Tensor3D<double> output;
		// Gradient with respect to the input of the layer. 
		Tensor3D<double> grad_input;
		// Name of the layer, used for convenience.
		std::string name;
		// Specific type of the layer.
		LayerType type;
	};
}