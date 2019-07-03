#pragma once
#include <vector>
#include <memory>
#include "Layer.h"
#include "Conv.h"
#include "ReLU.h"
#include "MaxPool.h"
#include "FC.h"
#include "Softmax.h"

namespace convnet_core {
	typedef std::unique_ptr<layer::Layer> Layer_ptr;
	
	class Model
	{
	public:
		Model();
		~Model();

		void Fit(Tensor3D<double>& input, Tensor3D<double>& target);
		Tensor3D<double> Predict(Tensor3D<double>& input);
		void Save(std::string path);
		void Load(std::string path);

		void Add(layer::Conv& layer);
		void Add(layer::ReLU& layer);
		void Add(layer::MaxPool& layer);
		void Add(layer::FC& layer);
		void Add(layer::Softmax& layer);

		std::vector<layer::Layer*> layers;
	private:
		//std::vector<Layer_ptr> layers;
	};
}

