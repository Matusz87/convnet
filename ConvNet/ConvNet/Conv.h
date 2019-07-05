// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#pragma once
#include "Layer.h"

namespace layer {
	// Convolutional layer. Applies trained filters on a tensor.
	class Conv : public Layer
	{
	public:
		Conv();
		~Conv();
		Conv(int height, int width, int depth, std::string name, 
			 int f_count, int f_size, int stride, int padding);
		Conv(const Conv & other);
		Conv(convnet_core::Triplet shape, std::string name,
			 int f_count, int f_size, int stride, int padding);
		Conv(convnet_core::Tensor3D<double>& prev_activation, std::string name,
			 int f_count, int f_size, int stride, int padding);

		// Slides filters over the input and performs convolution.
		void Forward(const Tensor3D<double>& prev_activation) override;
		// Calculates gradients based on the upstream gradient.
		void Backprop(Tensor3D<double>& grad_output) override;
		// Adjudsts weights based on the obtained gradients.
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		// Used for model saving.
		nlohmann::json Serialize() override;
		// Not implemented.
		double Loss(Tensor3D<double>& target) override;

		// Getter methods.
		std::vector<Tensor3D<double>>& GetWeights();
		std::vector<Tensor3D<double>>& GetBias();
		std::vector<Tensor3D<double>>& GetGradWeights();
		std::vector<Tensor3D<double>>& GetGradBias();
		Tensor3D<double> GetGradInput();
		
		// Getters for serialization.
		int GetFilterCount();
		int GetFilterSize();
		int GetStride();
		int GetPadding();

	private:
		// Number of filters of layer (i.e. output depth).
		int filter_count;
		// Number of filter size. Filters must be squares.
		int filter_size;
		// "Step size" of filter when we slide the filter.
		int stride;
		// Number of zeros around the border.
		int padding;

		// Vector of weight tensor. 
		// Shape: (f_count, f_size, f_size, inp_depth).
		std::vector<Tensor3D<double>> weights;
		// Vector of bias. Shape: (f_count, 1, 1, 1).
		std::vector<Tensor3D<double>> bias;
		// Gradients of weights w.r.t. error from prev layer.
		std::vector<Tensor3D<double>> grad_weights;
		// Gradients of bias  w.r.t. error from prev layer.
		std::vector<Tensor3D<double>> grad_bias;
		// Velocities for Nesterov Accelerated Gradient.
		std::vector<Tensor3D<double>> velocities;

		// Initializer methods for weights, biases, gradients.
		void InitWeights();
		void InitBias();
		void InitGrads();
		// Add a border with zeros to the tensor.
		// Applied along the depth dimension 
		// (i.e. zero-pad every matrix in the tensor).
		Tensor3D<double> ZeroPad(Tensor3D<double> tensor);
		// Removes previously added padding from a tensor (along depth dimension).
		Tensor3D<double> Unpad(Tensor3D<double> tensor);
	};
}
