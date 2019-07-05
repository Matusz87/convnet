// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#include "Model.h"
#include "Conv.h"
#include "Utils.h"

namespace convnet_core {
	// Default constructor and destructor.
	Model::Model()  { }
	Model::~Model() { }

	// Fits the model with an image. Includes forward pass and backpropagation.
	// Finnaly, trainable parameters are updated based on the gradients obtained
	// by the backprop phase.
	// @param input:	input image as a 3D tensor.
	// @param target:	one-hot encoded target variable. In traffic sign, it is (12, 1, 1)D vector
	// @param lr:		learning rate hyperparameter, used for weight update.
	// @param momentum: Nesterov momentum, used for ensuring faster convergence.
	// @returns	pair<bool, double>: a pair that contains whether prediction was correct and the loss.
	std::pair<bool, double> Model::Fit(Tensor3D<double>& input, 
									   Tensor3D<double>& target,
									   double lr, double momentum) {
		// Stores whether prediction was correct and the loss.
		bool correct = false;
		double loss = 1000;
		Tensor3D<double> predicted = Predict(input);

		if (utils::ComparePrediction(predicted, target))
			correct = true;

		Tensor3D<double> error = (predicted - target);
		loss = layers.back()->Loss(target);

		// Backpropagation to calculate the gradients.
		for (int i = layers.size()-1; i >= 0; --i) {
			if (i == layers.size() - 1) {
				layers[i]->Backprop(error);
			} else {
				// Handle flattened FC inputs.
				if (layers[i+1]->GetType() == layer::LayerType::FC) {
					Tensor3D<double> reshaped = layers[i+1]->GetGrads().Reshape(layers[i]->GetOutputShape());
					layers[i]->Backprop(reshaped);
				} else {
					layers[i]->Backprop(layers[i + 1]->GetGrads());
				}				
			}
				
			// Adjust weights based on the calculated gradients and learning rate.
			layers[i]->UpdateWeights(lr);
		}

		return std::pair<bool, double>(correct, loss);
	}

	// Classifies an image represented as a 3D tensor.
	// @param input:	input image as a 3D tensor.
	// @returns:		one-hot encoded prediction.
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

	// Saves a model to the hard disk.
	// @param path: path of the model file.
	void Model::Save(std::string path) {
		nlohmann::json model_json;
		for (int i = 0; i < layers.size(); ++i) {
			char c = IntToAlphabet(i);
			model_json["layer_"+c] = layers[i]->Serialize();
		}

		std::ofstream o(path);
		o << std::setw(4) << model_json;
	}

	// Loads a model to the hard disk.
	// @param path: path of the model file.
	void Model::Load(std::string path) {
		std::ifstream i(path);
		nlohmann::json model_json;
		i >> model_json;

		// Construct layers from the model file.
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

	// Evaluates an image an returns whether prediction was accurate and with the loss.
	// @param input:	input image as a 3D tensor.
	// @param target:	one-hot encoded target variable.
	// @returns	pair<bool, double>: a pair that contains whether prediction was correct and the loss.
	std::pair<bool, double> Model::Evaluate(Tensor3D<double>& input, Tensor3D<double>& target) {
		Tensor3D<double> predicted = Predict(input);
		bool correct = false;
		double loss;

		if (utils::ComparePrediction(predicted, target))
			correct = true;

		loss = layers.back()->Loss(target);

		
		return std::pair<bool, double>(correct, loss);
	}

	// Maps int numbers to the alphabet.
	// Required for serialization.
	// param n: number to map
	char Model::IntToAlphabet(int n) {
		assert(n >= 1 && n <= 26);

		return "abcdefghijklmnopqrstuvwxyz"[n - 1];
	}
}

