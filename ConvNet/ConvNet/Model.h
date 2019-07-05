// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

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
	// This class is responsible for representing a CNN.
	// Interface is inspired by Keras.
	class Model
	{
	public:
		Model();
		~Model();

		// Fits the model with one training example.
		std::pair<bool, double> Fit(Tensor3D<double>& input, 
									Tensor3D<double>& target,
									double learning_rate, double momentum=0.9);
		// Classifies an image. 
		Tensor3D<double> Predict(const Tensor3D<double>& input);
		// Saves a trained model.
		void Save(std::string path);
		// Loads a model from hard disk.
		void Load(std::string path);
		// Adds a layer to the layer container.
		void Add(layer::Layer* layer);
		// Evaluates an image an returns whether prediction was accurate and with the loss.
		std::pair<bool, double> Evaluate(Tensor3D<double>& input, Tensor3D<double>& target);

	private:
		// Container of layers, layers are dynamically typed in order to 
		// apply specialized methods.
		std::vector<layer::Layer*> layers;
	};
}

