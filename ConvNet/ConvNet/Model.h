#pragma once
#include <vector>
#include <memory>
#include "Layer.h"

namespace convnet_core {
	typedef std::unique_ptr<layer::Layer> Layer_ptr;
	
	class Model
	{
	public:
		Model();
		~Model();

		void Fit(Tensor3D<double> input, Tensor3D<double> target);
		void Predict(Tensor3D<double> input);
		void Add(Layer_ptr layer, std::string type, std::string name,
			int height, int width, int depth, int f_count = 0,
			int f_size = 0, int stride = 1, int padding = 0,
			int pool_size = 0, int in = 0, int out = 0);

	private:
		std::vector<Layer_ptr> layers;
		Tensor3D<double> input;
		Tensor3D<double> target;
	};
}

