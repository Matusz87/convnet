#pragma once
#include "Layer.h"

namespace layer {
	class FC : public Layer
	{
	public:
		FC();
		~FC();

		FC(std::string name, int num_hidden);
		FC(std::string name, int num_input, int num_hidden);
		FC(convnet_core::Triplet shape, std::string name, int num_hidden);
		FC(convnet_core::Tensor3D<double>& prev_activation, 
		   std::string name, int num_hidden);

		void Forward(Tensor3D<double> prev_activation) override;
		void Backprop(Tensor3D<double> grad_output) override;
		//Tensor3D<double> ZeroPad(Tensor3D<double> tensor);

		// Only for testing purposes.
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
		// Auxiliary container for backprop.
		std::vector<double> output_values;

		void InitWeights();
		void InitBias();
		void InitGrads();
		int MapToUnrolledIndex(int row, int col, int depth);
	};
}
