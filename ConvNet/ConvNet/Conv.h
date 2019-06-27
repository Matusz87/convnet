#pragma once
#include "Layer.h"

namespace layer {
	class Conv : public Layer
	{
	public:
		Conv();
		~Conv();
		Conv(convnet_core::Triplet shape, std::string name,
			 int f_count, int f_size, int stride, int padding);
		Conv(convnet_core::Tensor3D<double>& prev_activation, std::string name,
			 int f_count, int f_size, int stride, int padding);

		void Forward(Tensor3D<double> prev_activation) override;
		void Backprop(Tensor3D<double> grad_output) override;
		//Tensor3D<double> ZeroPad(Tensor3D<double> tensor);

		// Only for testing purposes.
		std::vector<Tensor3D<double>>& GetWeights();
		std::vector<Tensor3D<double>>& GetBias();

	private:
		// Number of filters of layer (i.e. output depth).
		int filter_count;
		int filter_size;
		int stride;
		int padding;
		std::vector<Tensor3D<double>> weights;
		std::vector<Tensor3D<double>> bias;
		std::vector<Tensor3D<double>> grad_weights;
		std::vector<Tensor3D<double>> grad_bias;

		void InitWeights();
		void InitBias();
		void InitGrads();
		Tensor3D<double> ZeroPad(Tensor3D<double> tensor);
	};
}