// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#pragma once
#include "Layer.h"

namespace layer {
	// Fully-connected layer.
	class FC : public Layer
	{
	public:
		FC();
		~FC();

		FC(std::string name, int num_hidden);
		FC(std::string name, int num_input, int num_hidden);
		FC(convnet_core::Triplet shape, std::string name, int num_hidden);
		FC(const FC & other);
		FC(convnet_core::Tensor3D<double>& prev_activation,
		   std::string name, int num_hidden);
 
		// Forward pass.
		void Forward(const Tensor3D<double>& prev_activation) override;
		// Calculates gradients based on the upstream gradient.
		void Backprop(Tensor3D<double>& grad_output) override;
		// Adjudsts weights based on the obtained gradients.
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		// Used for model saving.
		nlohmann::json Serialize() override;
		// Not implemented.
		double Loss(Tensor3D<double>& target) override;

		// Getter methods
		Tensor3D<double>& GetWeights();
		Tensor3D<double>& GetBias();
		Tensor3D<double>& GetGradWeights();
		Tensor3D<double>& GetGradBias();
		Tensor3D<double> GetGradInput();

	private:
		// Number of neurons in the output layer
		int num_hidden;
		// Vector of weight tensor. 
		// Shape: (input.height, output.height, 1).
		Tensor3D<double> weights;
		// Shape: (output.height, 1, 1).
		Tensor3D<double> bias;
		// Gradients of weights w.r.t. error from prev layer.
		Tensor3D<double> grad_weights;
		// Gradients of bias  w.r.t. error from prev layer.
		Tensor3D<double> grad_bias;
		// Velocities for momentum.
		Tensor3D<double> velocities;
		// Auxiliary tensor for flattening in the forward pass.
		Tensor3D<double> tmp_input;
		// Required for model loading.
		bool has_weights_initialized;

		// Initialization methods.
		void InitWeights();
		void InitBias();
		void InitGrads();
		// Returns with the index of data element in the unrolled vector.
		int MapToUnrolledIndex(int row, int col, int depth);
	};
}
