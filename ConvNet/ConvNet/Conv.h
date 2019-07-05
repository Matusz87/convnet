#pragma once
#include "Layer.h"

namespace layer {
	class Conv : public Layer
	{
	public:
		Conv();
		~Conv();
		Conv(int height, int width, int depth, std::string name, int f_count, int f_size, int stride, int padding);
		Conv(const Conv & other);
		Conv(convnet_core::Triplet shape, std::string name,
			 int f_count, int f_size, int stride, int padding);
		Conv(convnet_core::Tensor3D<double>& prev_activation, std::string name,
			 int f_count, int f_size, int stride, int padding);

		void Forward(const Tensor3D<double>& prev_activation) override;
		void Backprop(Tensor3D<double>& grad_output) override;
		void UpdateWeights(double learning_rate, double momentum = 0.9) override;
		nlohmann::json Serialize() override;
		double Loss(Tensor3D<double>& target) override;

		std::vector<Tensor3D<double>>& GetWeights();
		std::vector<Tensor3D<double>>& GetBias();
		std::vector<Tensor3D<double>>& GetGradWeights();
		std::vector<Tensor3D<double>>& GetGradBias();
		Tensor3D<double> GetGradInput();
		/*Tensor3D<double> Unpad(Tensor3D<double> tensor);
		Tensor3D<double> ZeroPad(Tensor3D<double> tensor);*/

		// Getters for serializing.
		int GetFilterCount();
		int GetFilterSize();
		int GetStride();
		int GetPadding();

	private:
		// Number of filters of layer (i.e. output depth).
		int filter_count;
		int filter_size;
		int stride;
		int padding;
		std::vector<Tensor3D<double>> weights;
		std::vector<Tensor3D<double>> bias;
		// Gradients of weights w.r.t. error from prev layer.
		std::vector<Tensor3D<double>> grad_weights;
		// Gradients of bias  w.r.t. error from prev layer.
		std::vector<Tensor3D<double>> grad_bias;
		// Velocities for momentum.
		std::vector<Tensor3D<double>> velocities;

		void InitWeights();
		void InitBias();
		void InitGrads();
		Tensor3D<double> ZeroPad(Tensor3D<double> tensor);
		Tensor3D<double> Unpad(Tensor3D<double> tensor);
	};
}
