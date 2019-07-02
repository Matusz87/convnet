#include "Model.h"

namespace convnet_core {
	Model::Model()  { }
	Model::~Model() { }

	void Model::Fit(Tensor3D<double> input, Tensor3D<double> target) {
	}

	void Model::Predict(Tensor3D<double> input) {
	}

	void Model::Add(Layer_ptr layer, std::string type, std::string name,
					int height, int width, int depth, int f_count,
					int f_size, int stride, int padding, 
					int pool_size, int in, int out) {
		// TODO: switch based on the layer type
		//layers.push_back(layer);
	}
}

