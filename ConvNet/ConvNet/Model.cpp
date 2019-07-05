#include "Model.h"
#include "Conv.h"
#include "Utils.h"

namespace convnet_core {
	Model::Model()  { }
	Model::~Model() { }

	std::pair<bool, double> Model::Fit(Tensor3D<double>& input, 
									   Tensor3D<double>& target,
									   double lr, double momentum) {
		// Stores whether prediction was correct and the loss.
		bool correct;
		double loss;
		Tensor3D<double> predicted = Predict(input);

		if (utils::ComparePrediction(predicted, target))
			++correct;

		Tensor3D<double> error = (predicted - target);
		loss = layers.back()->Loss(target);

		// Backpropagation to calculate the gradients.
		for (int i = layers.size()-1; i >= 0; --i) {
			if (i == layers.size() - 1) {
				layers[i]->Backprop(error);
			} else {
				// Handle flattened FC inputs.
				if (layers[i+1]->GetType() == layer::LayerType::FC) {
//					std::cout << "Reshape FC grad" << std::endl;
					Tensor3D<double> reshaped = layers[i+1]->GetGrads().Reshape(layers[i]->GetOutputShape());
					layers[i]->Backprop(reshaped);
//					std::cout << "Bacprop after FC" << std::endl;
				}
				else {
//					std::cout << "Other backprop" << std::endl;
					layers[i]->Backprop(layers[i + 1]->GetGrads());
				}				
			}
				
			// Adjust weights based on the calculated gradients and learning rate.
			layers[i]->UpdateWeights(lr);
		}

		return std::pair<bool, double>(correct, loss);
	}

	Tensor3D<double> Model::Predict(const Tensor3D<double>& input) {
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
		std::ifstream i(path);
		nlohmann::json model_json;
		i >> model_json;

		for (auto& layer : model_json) {
			if (layer["type"] == "pool") {
				layer::MaxPool* pool = new layer::MaxPool(utils::ReadPoolLayerJSON(layer));
				Add(pool);
			} else if (layer["type"] == "conv") {
				layer::Conv *conv = new layer::Conv(utils::ReadConvLayerJSON(layer));
				Add(conv);
			} else if (layer["type"] == "relu") {
				layer::ReLU* relu = new layer::ReLU(utils::ReadReLUJSON(layer));
				Add(relu);
			} else if (layer["type"] == "fc") {
				layer::FC *fc = new layer::FC(utils::ReadFCLayerJSON(layer));
				Add(fc);
			} else if (layer["type"] == "softmax") {
				layer::Softmax* softmax = new layer::Softmax(utils::ReadSoftmaxJSON(layer));
				Add(softmax);
			}
		}
		
	}

	void Model::Add(layer::Layer* layer) {
		layers.push_back(layer);
	}

	std::pair<bool, double> Model::Evaluate(Tensor3D<double>& input, Tensor3D<double>& target) {
		Tensor3D<double> predicted = Predict(input);
		bool correct;
		double loss;

		if (utils::ComparePrediction(predicted, target))
			++correct;

		loss = layers.back()->Loss(target);

		
		return std::pair<bool, double>(correct, loss);
	}

	//void Model::Add(layer::Conv& layer) {
	//	layer::Layer* l;
	//	l = &layer;
	//	layers.push_back(l);
	//}
	//
	//void Model::Add(layer::ReLU& layer) {
	//	layer::Layer* l;
	//	l = &layer;
	//	layers.push_back(l);
	//}
	//void Model::Add(layer::MaxPool& layer) {
	//	layer::Layer* l;
	//	l = &layer;
	//	layers.push_back(l);
	//}
	//void Model::Add(layer::FC& layer) {
	//	layer::Layer* l;
	//	l = &layer;
	//	layers.push_back(l);
	//}
	//void Model::Add(layer::Softmax& layer) {
	//	layer::Layer* l;
	//	l = &layer;
	//	layers.push_back(l);
	//}
}

