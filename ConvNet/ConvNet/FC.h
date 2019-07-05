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
 
		// Forward pass, 
		void Forward(const Tensor3D<double>& prev_activation) override;
		// Calculates gradients based on the upstream gradient.
		void Backprop(Tensor3D<double>& grad_output) override;
		// Adjudsts weights based on the obtained gradients.
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		// Used for model saving.
		nlohmann::json Serialize() override;
		// Empty.
		double Loss(Tensor3D<double>& target) override;

		// Getter methods
		Tensor3D<double>& GetWeights();
		Tensor3D<double>& GetBias();
		Tensor3D<double>& GetGradWeights();
		Tensor3D<double>& GetGradBias();
		Tensor3D<double> GetGradInput();

	private:
		// Number of filters of layer (i.e. output depth).
		int num_hidden;
		Tensor3D<double> weights;
		Tensor3D<double> bias;
		// Gradients of weights w.r.t. error from prev layer.
		Tensor3D<double> grad_weights;
		// Gradients of bias  w.r.t. error from prev layer.
		Tensor3D<double> grad_bias;
		// Velocities for momentum.
		Tensor3D<double> velocities;
		// Auxiliary tensor for flattening in the forward pass.
		Tensor3D<double> tmp_input;
		bool has_weights_initialized;

		void InitWeights();
		void InitBias();
		void InitGrads();
		int MapToUnrolledIndex(int row, int col, int depth);
	};
}
