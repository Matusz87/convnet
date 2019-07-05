// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#pragma once
#include "Layer.h"

namespace layer {
	// Rectified linear unit, non-linearity layer.
	class ReLU : public Layer
	{
	public:
		ReLU();
		ReLU(std::string name, int height, int width, int depth);
		ReLU(const ReLU & other);
		ReLU(convnet_core::Triplet shape, std::string name);
		ReLU(Tensor3D<double>& prev_activation, std::string name);
		~ReLU();

		// Forward pass.
		void Forward(const Tensor3D<double>& prev_activation) override;
		// Calculates gradients based on the upstream gradient.
		void Backprop(Tensor3D<double>& grad_out) override;
		// Not implemented, no trainable params.
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		// Used for model saving.
		nlohmann::json Serialize() override;
		// Not implemented.
		double Loss(Tensor3D<double>& target) override;
	};
}


