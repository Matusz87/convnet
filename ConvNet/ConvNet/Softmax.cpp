#include "Softmax.h"
#include <math.h>

namespace layer {

	Softmax::Softmax() { }
	Softmax::~Softmax() { }

	Softmax::Softmax(std::string name, int height, int width, int depth)
				: Layer(name) {
		input = Tensor3D<double>(height, width, depth);
		output = Tensor3D<double>(height, width, depth);
		grad_input = Tensor3D<double>(height, width, depth);
	}

	Softmax::Softmax(convnet_core::Tensor3D<double>& prev_activation,
		std::string name) : Layer(prev_activation, name) {
		convnet_core::Triplet shape = prev_activation.GetShape();
		output = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
	}

	void Softmax::Forward(Tensor3D<double> prev_activation) {
		assert(prev_activation.GetShape().width == 1 &&
			prev_activation.GetShape().depth == 1);

		double sum_exp = 0;
		double exp_val = 0;
		for (int i = 0; i < prev_activation.GetShape().height; ++i) {
			exp_val = exp(prev_activation(i, 0, 0));
			output(i, 0, 0) = exp_val;
			sum_exp += exp_val;
		}

		output = output / sum_exp;
	}

	// Cross-entropy loss function.
	double Softmax::Loss(Tensor3D<double> target) {
		assert(target.GetShape().height == output.GetShape().height);

		for (int i = 0; i < output.GetShape().height; ++i) {
			if (target(i, 0, 0) == 1)
				return -log(output(i, 0, 0)) * -1;
		}
		
	}

	void Softmax::Backprop(Tensor3D<double> grad_out) {
		grad_input = grad_out;
	}

}
