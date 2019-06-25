#pragma once
#include "Layer.h"

namespace layer {
	class ReLU : public Layer
	{
	public:
		ReLU();
		ReLU(convnet_core::Shape shape, std::string name);
		ReLU(convnet_core::Tensor3D<double>& prevAct, std::string name);
		~ReLU();

		void Forward(Tensor3D<double> prevAct) override;
		void Backprop(Tensor3D<double> dOut) override;
	};
}


