#pragma once
#include "Layer.h"
namespace layer {
	class Softmax : public Layer
	{
	public:
		Softmax();
		~Softmax();

		Softmax(convnet_core::Tensor3D<double>& prev_activation, std::string name);
		Softmax(std::string name, int height, int width, int depth);

		void Forward(Tensor3D<double> prev_activation) override;
		double Loss(Tensor3D<double> target);
		void Backprop(Tensor3D<double> grad_out) override;
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
	};

}
