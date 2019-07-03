#include "Model.h"
#include "Conv.h"
#include "Utils.h"

namespace convnet_core {
	Model::Model()  { }
	Model::~Model() { }

	void Model::Fit(Tensor3D<double>& input, Tensor3D<double>& target) {
	}

	Tensor3D<double> Model::Predict(Tensor3D<double>& input) {
		for (int i = 0; i < layers.size(); ++i) {
			if (i == 0) {
					layers[i]->Forward(input);
			} else {
				layers[i]->Forward(layers[i-1]->GetOutput());
			}
		}

		return layers.back()->GetOutput();
	}

	void Model::Save(std::string path) {
		nlohmann::json model_json;
		for (int i = 0; i < layers.size(); ++i) {
			model_json["layer_"+std::to_string(i)] = layers[i]->Serialize();
		}

		std::ofstream o(path);
		o << std::setw(4) << model_json;
	}

	void Model::Load(std::string path) {
	}

	void Model::Add(layer::Conv& layer) 	{
		layer::Layer* l;
		l = &layer;
		layers.push_back(l);
	}
	
	void Model::Add(layer::ReLU& layer) {
		layer::Layer* l;
		l = &layer;
		layers.push_back(l);
	}
	void Model::Add(layer::MaxPool& layer) {
		layer::Layer* l;
		l = &layer;
		layers.push_back(l);
	}
	void Model::Add(layer::FC& layer) {
		layer::Layer* l;
		l = &layer;
		layers.push_back(l);
	}
	void Model::Add(layer::Softmax& layer) {
		layer::Layer* l;
		l = &layer;
		layers.push_back(l);
	}
}

