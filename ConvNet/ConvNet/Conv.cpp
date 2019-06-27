#include "Conv.h"


namespace layer {
	Conv::Conv() { }


	Conv::~Conv() { }

	Conv::Conv(convnet_core::Triplet shape, std::string name, int f_count, 
			   int f_size, int stride, int padding) : Layer(shape, name) {
		filter_count = f_count;
		filter_size = f_size;
		this->stride = stride;
		this->padding = padding;

		int out_height = ((shape.height - f_size + 2 * padding) / stride) + 1;
		int out_width = ((shape.width - f_size + 2 * padding) / stride) + 1;
		output = Tensor3D<double>(out_height, out_width, f_count);
		
		// Init variables.
		InitWeights();
		InitBias();
		InitGrads();
	}

	Conv::Conv(convnet_core::Tensor3D<double>& prev_activation, std::string name, 
			   int f_count, int f_size, int stride, int padding)
		: Layer(prev_activation, name) {
		filter_count = f_count;
		filter_size = f_size;
		this->stride = stride;
		this->padding = padding;

		convnet_core::Triplet shape = prev_activation.GetShape();
		int out_height = ((shape.height - f_size + 2 * padding) / stride) + 1;
		int out_width = ((shape.width - f_size + 2 * padding) / stride) + 1;
		output = Tensor3D<double>(out_height, out_width, f_count);

		// Init variables.
		InitWeights();
		InitBias();
		InitGrads();
	}

	void Conv::Forward(Tensor3D<double> prev_activation) {
		Tensor3D<double> padded = ZeroPad(input);
		convnet_core::Triplet out_shape = GetOutputShape();

		int vert_start, vert_end, horiz_start, horiz_end;
		double inp;	// Value from input tensor.
		int sum = 0;
		// Row / column indexes for weight. Auxiliary variables for convolution.
		int w_row = 0; int w_col = 0; 

		for (int c = 0; c < out_shape.depth; ++c) {
			Tensor3D<double> W = weights[c];
			Tensor3D<double> b = bias[c];

			for (int h = 0; h < out_shape.height; ++h) {
				for (int w = 0; w < out_shape.width; ++w) {
					//for (int c = 0; c < out_shape.depth; ++c) {
						// Calculate the boundary indexes of the slice.
						vert_start = h * stride;
						vert_end = vert_start + filter_size;
						horiz_start = w * stride;
						horiz_end = horiz_start + filter_size;
//std::cout << "v_s: " << vert_start << ", v_e: " << vert_end << ", h_s: " << horiz_start << ", h_e: " << horiz_end << std::endl;
						sum = 0;
						
						// Dot product between input-slice and filter along the depth dimension.
						for (int weight_depth = 0; weight_depth < W.GetShape().depth; ++weight_depth) {
							w_row = 0;
							for (int v_slice = vert_start; v_slice < vert_end; ++v_slice) {
								w_col = 0;
								for (int h_slice = horiz_start; h_slice < horiz_end; ++h_slice) {
									double w = W(w_row, w_col, weight_depth);									
									inp = padded(v_slice, h_slice, weight_depth);
//std::cout << "v_s: " << v_slice << ", h_s: " << h_slice << ", w_d: " << weight_depth<< std::endl;
//std::cout << "w_r: " << w_row << ", w_c: " << w_col << std::endl;
//std::cout << "I: " << inp << ", W: " << w << std::endl;
									sum += w * inp;

									++w_col;
								}
								++w_row;
							}								
						}
						output(h, w, c) = sum + b(0,0,0);
					}
				}
			}
	}

	void Conv::Backprop(Tensor3D<double> grad_output) {

	}

	std::vector<Tensor3D<double>>& Conv::GetWeights() {
		return weights;
	}

	std::vector<Tensor3D<double>>& Conv::GetBias() {
		return bias;
	}

	void Conv::InitWeights() {
		for (int i = 0; i < filter_count; ++i) {
			Tensor3D<double> t(filter_size, filter_size, input.GetShape().depth);
			t.InitRandom();
			weights.push_back(t);
		}
	}

	void Conv::InitBias() {
		for (int i = 0; i < filter_count; ++i) {
			Tensor3D<double> t(1, 1, 1);
			t.InitZeros();
			bias.push_back(t);
		}
	}

	void Conv::InitGrads() {
		grad_input = Tensor3D<double>(input.GetShape());
		grad_input.InitZeros();

		for (int i = 0; i < filter_count; ++i) {
			Tensor3D<double> dB(1, 1, 1);
			dB.InitZeros();
			grad_bias.push_back(dB);

			Tensor3D<double> dW(filter_size, filter_size, input.GetShape().depth);
			dW.InitZeros();
			grad_weights.push_back(dW);
		}
	}

	Tensor3D<double> Conv::ZeroPad(Tensor3D<double> tensor) {
		convnet_core::Triplet shape = tensor.GetShape();
		Tensor3D<double> padded(shape.height + 2 * padding,
								shape.width + 2 * padding,
								shape.depth);
		padded.InitZeros();

		for (int i = 0; i < shape.height; ++i) {
			for (int j = 0; j < shape.width; ++j) {
				for (int k = 0; k < shape.depth; ++k) {
					padded(i + padding, j + padding, k) = tensor(i, j, k);
				}
			}
		}

		return padded;
	}
}
