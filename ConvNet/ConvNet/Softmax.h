// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#pragma once
#include "Layer.h"
namespace layer {
	// Softmax non-linearity, applied on the output of last FC layer.
	// Calculates the probabilities of belonging to each class.
	class Softmax : public Layer
	{
	public:
		Softmax();
		~Softmax();

		Softmax(convnet_core::Tensor3D<double>& prev_activation, std::string name);
		Softmax(std::string name, int height, int width, int depth);
		Softmax(const Softmax & other);

		// Forward pass, calculates softmax function on each data element.
		void Forward(const Tensor3D<double>& prev_activation) override;
		// Calculates the categorical cross entropy loss w.r.t. to an input/target pair.
		double Loss(Tensor3D<double>& target) override;
		// Calculates gradients based on the upstream gradient.
		void Backprop(Tensor3D<double>& grad_out) override;
		// Not implemented, no trainable parameters.
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		// Used for model saving.
		nlohmann::json Serialize() override;
	};

}
