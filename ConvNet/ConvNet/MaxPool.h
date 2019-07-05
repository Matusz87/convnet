// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#pragma once
#include "Layer.h"
#include <nlohmann\json.hpp>

namespace layer {
	// MaxPool Layer which is responsible for spatially reducing layers.
	// Also used for ensuring translation invariance.
	class MaxPool : public Layer {
	public:
		MaxPool();
		~MaxPool();
		MaxPool(std::string name, int height, int width, int depth,
				int stride, int pool_size);
		MaxPool(convnet_core::Triplet shape, std::string name,
			int stride, int pool_size);
		MaxPool(convnet_core::Tensor3D<double>& prev_activation, std::string name,
			int stride, int pool_size);
		MaxPool(const MaxPool& other);

		// Reduces the spatial size of input.
		void Forward(const Tensor3D<double>& prev_activation) override;
		// Calculates gradients based on the upstream gradient.
		void Backprop(Tensor3D<double>& grad_output) override;
		// Not implemented, there are no trainable parameters of MaxPool layer.
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		// Used for model saving.
		nlohmann::json Serialize() override;
		// Not implemented.
		double Loss(Tensor3D<double>& target) override;

		// Getters for serialization.
		int GetPoolSize();
		int GetStride();

	private:
		// "Step size" of filter when we slide the window.
		int stride;
		// Size of subsampling windows.
		int pool_size;
		// Stores the indexes of max element in each slices.
		// Used in backprop for gradient routing.
		std::vector<convnet_core::Triplet> max_indexes;
	};
}


