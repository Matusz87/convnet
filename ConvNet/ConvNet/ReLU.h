#pragma once
#include "Layer.h"

namespace layer {
	class ReLU : public Layer
	{
	public:
		ReLU();
		ReLU(std::string name, int height, int width, int depth);
		ReLU(convnet_core::Triplet shape, std::string name);
		ReLU(convnet_core::Tensor3D<double>& prev_activation, std::string name);
		~ReLU();

		void Forward(Tensor3D<double> prev_activation) override;
		void Backprop(Tensor3D<double> grad_out) override;
	};
}


