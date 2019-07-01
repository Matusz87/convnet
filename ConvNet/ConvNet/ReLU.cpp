#include "ReLU.h"

namespace layer {
	ReLU::ReLU() { 	}

	ReLU::ReLU(std::string name, int height, int width, int depth) : Layer(name)	{
		input = Tensor3D<double>(height, width, depth);
		output = Tensor3D<double>(height, width, depth);
		grad_input = Tensor3D<double>(height, width, depth);
	}

	ReLU::ReLU(convnet_core::Triplet shape, std::string name) : Layer(shape, name) {
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
	}

	ReLU::ReLU(convnet_core::Tensor3D<double>& prev_activation, std::string name) 
			: Layer(prev_activation, name) {
		convnet_core::Triplet shape = prev_activation.GetShape();
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
	}

	void ReLU::Forward(Tensor3D<double> prev_activation) {
		input = Tensor3D<double>(prev_activation);
		
		int depth = input.GetShape().depth;
		int height = input.GetShape().height;
		int width = input.GetShape().width;
		output = Tensor3D<double>(height, width, depth);

		for (int k = 0; k < depth; ++k) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					output(i, j, k) = input(i, j, k) > 0 ? input(i, j, k) : 0;
				}
			}
		}
	}

	//https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/relu_layer.html
	void ReLU::Backprop(Tensor3D<double> grad_out) {
		grad_input.InitZeros();

		int depth = input.GetShape().depth;
		int height = input.GetShape().height;
		int width = input.GetShape().width;
		for (int k = 0; k < depth; ++k) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					grad_input(i, j, k) = input(i, j, k) < 0 ? 0 : (1 * grad_out(i, j, k));
				}
			}
		}
	}

	void ReLU::UpdateWeights(double lr, double momentum) { }

	ReLU::~ReLU() { }
}


