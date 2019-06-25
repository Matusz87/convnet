#include "ReLU.h"

namespace layer {
	ReLU::ReLU() { 	}

	ReLU::ReLU(convnet_core::Shape shape, std::string name) : Layer(shape, name) {
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		dInput = Tensor3D<double>(shape.height, shape.width, shape.depth);

		std::cout << "Output shape: ";
		convnet_core::PrintShape(GetOutputShape());
		std::cout << "Grads shape: ";
		convnet_core::PrintShape(GetGradsShape());
	}

	ReLU::ReLU(convnet_core::Tensor3D<double>& prevAct, std::string name) : Layer(prevAct, name) {
		convnet_core::Shape shape = prevAct.GetShape();
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		dInput = Tensor3D<double>(shape.height, shape.width, shape.depth);

		std::cout << "Output shape: ";
		convnet_core::PrintShape(GetOutputShape());
		std::cout << "Grads shape: ";
		convnet_core::PrintShape(GetGradsShape());
	}

	void ReLU::Forward(Tensor3D<double> prevAct) {
		std::cout << "Input shape: ";
		convnet_core::PrintShape(prevAct.GetShape());

		int depth = input.GetShape().depth;
		int height = input.GetShape().height;
		int width = input.GetShape().width;
		for (int k = 0; k < depth; ++k) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					output(i, j, k) = input(i, j, k) > 0 ? input(i, j, k) : 0;
				}
			}
		}
	}

	//TODO: TEST IT!!!
	//https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/relu_layer.html
	void ReLU::Backprop(Tensor3D<double> dOut) {
		std::cout << "Input shape: ";
		convnet_core::PrintShape(dOut.GetShape());

		int depth = input.GetShape().depth;
		int height = input.GetShape().height;
		int width = input.GetShape().width;
		for (int k = 0; k < depth; ++k) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					output(i, j, k) = input(i, j, k) < 0 ? 0 : (1 * dInput(i, j, k));
				}
			}
		}
	}

	ReLU::~ReLU() { }
}


