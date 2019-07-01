#pragma once
#include "Layer.h"

namespace layer {
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

		void Forward(Tensor3D<double> prev_activation) override;
		void Backprop(Tensor3D<double> grad_output) override;
		void UpdateWeights(double learning_rate) override;

	private:
		int stride;
		int pool_size;
		// Stores the indexes of max element in each slices.
		// Used in backprop for gradient routing.
		std::vector<convnet_core::Triplet> max_indexes;
	};
}


