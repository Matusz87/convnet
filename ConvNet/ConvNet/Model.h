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

		std::pair<bool, double> Fit(Tensor3D<double>& input, 
									Tensor3D<double>& target,
									double learning_rate, double momentum=0.9);
		Tensor3D<double> Predict(const Tensor3D<double>& input);
		void Save(std::string path);
		void Load(std::string path);
		void Add(layer::Layer* layer);
		std::pair<bool, double> Evaluate(Tensor3D<double>& input, Tensor3D<double>& target);

	private:
		std::vector<layer::Layer*> layers;
	};
}

