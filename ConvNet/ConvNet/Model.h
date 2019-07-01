#pragma once
#include <vector>

#include "Layer.h"

namespace convnet_core {
	class Model
	{
	public:
		Model();
		~Model();

		void Fit(Tensor3D<double> input, Tensor3D<double> target);
		void Predict(Tensor3D<double> input);
		//void Add(layer::Layer layer);

	private:
		std::vector<layer::Layer> layers;
		Tensor3D<double> input;
		Tensor3D<double> target;
	};
}

