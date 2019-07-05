#pragma once
#include "Layer.h"

namespace layer {
	class ReLU : public Layer
	{
	public:
		ReLU();
		ReLU(std::string name, int height, int width, int depth);
		ReLU(const ReLU & other);
		ReLU(convnet_core::Triplet shape, std::string name);
		ReLU(Tensor3D<double>& prev_activation, std::string name);
		~ReLU();

		void Forward(const Tensor3D<double>& prev_activation) override;
		void Backprop(Tensor3D<double>& grad_out) override;
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		nlohmann::json Serialize() override;
		double Loss(Tensor3D<double>& target) override;
	};
}


