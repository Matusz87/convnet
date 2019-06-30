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
		double dotProduct = 0;
		// Row/column indexes for weight. Auxiliary variables for convolution.
		int w_row = 0; int w_col = 0; 

		// Loop over channels (filters).
		for (int c = 0; c < out_shape.depth; ++c) {
			Tensor3D<double> W = weights[c];
			Tensor3D<double> b = bias[c];

			// For each filter, convolve input with the filter.
			for (int h = 0; h < out_shape.height; ++h) {
				for (int w = 0; w < out_shape.width; ++w) {
					// Calculate the boundary indexes of the slice.
					vert_start = h * stride;
					vert_end = vert_start + filter_size;
					horiz_start = w * stride;
					horiz_end = horiz_start + filter_size;
//std::cout << "v_s: " << vert_start << ", v_e: " << vert_end << ", h_s: " << horiz_start << ", h_e: " << horiz_end << std::endl;
					dotProduct = 0;
						
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
								dotProduct += w * inp;
								++w_col;
							}
							++w_row;
						}								
					}
//std::cout << "h: " << h << ", w: " << w << ", c: " << c << ", dot: " << dotProduct << std::endl;
					output(h, w, c) = dotProduct + b(0,0,0);
				}
			}
		}
	}

	void Conv::Backprop(Tensor3D<double> grad_output) {
		// TODO: HANDLE PADDING!!!
		Tensor3D<double> padded = ZeroPad(input);
		grad_input.InitZeros();
		Tensor3D<double> grad_input_padded(padded.GetShape());
		grad_input_padded.InitZeros();

		convnet_core::Triplet out_shape = GetOutputShape();
		int vert_start, vert_end, horiz_start, horiz_end;
		double inp;	// Value from input tensor.
		double dotProduct = 0;
		// Sum of grads for grad_bias.
		double sum_dOut = 0;
		// Row/column indexes for weight. Auxiliary variables for convolution.
		int w_row = 0; int w_col = 0;

		// Loop over channels (filters).
		//for (int c = 0; c < out_shape.depth; ++c) {
		//	Tensor3D<double> W = weights[c];
		//	Tensor3D<double> b = bias[c];

		//	sum_dOut = 0;
		//	
		//	for (int h = 0; h < out_shape.height; ++h) {
		//		for (int w = 0; w < out_shape.width; ++w) {
		//			// Calculate the boundary indexes of the slice.
		//			vert_start = h * stride;
		//			vert_end = vert_start + filter_size;
		//			horiz_start = w * stride;
		//			horiz_end = horiz_start + filter_size;
		//			std::cout << "h: " << h << ", w: " << w << std::endl;// << ", h_s: " << horiz_start << ", h_e: " << horiz_end << std::endl;
		//			
		//			dotProduct = 0;
		//			// Dot product between input-slice and filter along the depth dimension.
		//			for (int weight_depth = 0; weight_depth < input.GetShape().depth; ++weight_depth) {
		//				w_row = 0;
		//				sum_dOut += grad_output(h, w, c);
		//				for (int v_slice = vert_start; v_slice < vert_end; ++v_slice) {
		//					w_col = 0;
		//					for (int h_slice = horiz_start; h_slice < horiz_end; ++h_slice) {
		//						std::cout << "v_s: " << v_slice << ", h_s: " << h_slice << ", w_d: " << weight_depth<< std::endl;
		//						std::cout << "w_r: " << w_row << ", w_c: " << w_col << std::endl;
		//						
		//						// Calculate grad_weights.
		//						// dW =  X * dOut
		//						inp = padded(v_slice, h_slice, weight_depth);
		//						double dO = grad_output(w_row, w_col, c);
		//						dotProduct += dO * inp;
		//						std::cout << "g.w.I: " << inp << ", W: " << dO << std::endl;

		//						// Calculate grad_input. "Full convolution" over weights.
		//						// dA += sum_h(sum_w(w x dOut_h_w)
		//						// where w is the weight and dOut_h_w is a scalar corresponding 
		//						// to the gradient of the cost with respect to the output.
		//						inp = W(w_row, w_col, weight_depth);
		//						dO = grad_output(h, w, c);
		//						//grad_input(v_slice, h_slice, weight_depth) += (inp * dO);
		//						grad_input_padded(v_slice, h_slice, weight_depth) += (inp * dO);
		//						std::cout << "g.i.I: " << inp << ", W: " << dO << std::endl << std::endl;
		//						
		//						++w_col;
		//					}
		//					++w_row;
		//				}
		//			}
		//			grad_weights[c](h, w, c) = dotProduct;
		//		}
		//	}
		//	grad_bias[c](0, 0, 0) = sum_dOut;
		//}
		//std::cout << "gradInputPadded: " << std::endl;
		//convnet_core::PrintTensor(grad_input_padded);
		//grad_input = Tensor3D<double>(Unpad(grad_input_padded));


		// Calculate gradients w.r.t. input and bias.
		for (int c = 0; c < out_shape.depth; ++c) {
			Tensor3D<double> W = weights[c];
			Tensor3D<double> b = bias[c];

			sum_dOut = 0;

			for (int h = 0; h < out_shape.height; ++h) {
				for (int w = 0; w < out_shape.width; ++w) {
					// Calculate the boundary indexes of the slice.
					vert_start = h * stride;
					vert_end = vert_start + filter_size;
					horiz_start = w * stride;
					horiz_end = horiz_start + filter_size;
					//std::cout << "h: " << h << ", w: " << w << std::endl;// << ", h_s: " << horiz_start << ", h_e: " << horiz_end << std::endl;

					dotProduct = 0;
					// Dot product between input-slice and filter along the depth dimension.
					for (int weight_depth = 0; weight_depth < input.GetShape().depth; ++weight_depth) {
						w_row = 0;
						sum_dOut += grad_output(h, w, c);
						for (int v_slice = vert_start; v_slice < vert_end; ++v_slice) {
							w_col = 0;
							for (int h_slice = horiz_start; h_slice < horiz_end; ++h_slice) {
								//std::cout << "v_s: " << v_slice << ", h_s: " << h_slice << ", w_d: " << weight_depth << std::endl;
								//std::cout << "w_r: " << w_row << ", w_c: " << w_col << std::endl;

								// Calculate grad_input. "Full convolution" over weights.
								// dA += sum_h(sum_w(w x dOut_h_w)
								// where w is the weight and dOut_h_w is a scalar corresponding 
								// to the gradient of the cost with respect to the output.
								inp = W(w_row, w_col, weight_depth);
								double dO = grad_output(h, w, c);
								//grad_input(v_slice, h_slice, weight_depth) += (inp * dO);
								grad_input_padded(v_slice, h_slice, weight_depth) += (inp * dO);
								//std::cout << "g.i.I: " << inp << ", W: " << dO << std::endl << std::endl;

								++w_col;
							}
							++w_row;
						}
					}
					//grad_weights[c](h, w, c) = dotProduct;
				}
			}
			grad_bias[c](0, 0, 0) = sum_dOut;
		}
		grad_input = Tensor3D<double>(Unpad(grad_input_padded));

		for (int c = 0; c < out_shape.depth; ++c) {
			Tensor3D<double> W = weights[c];
			Tensor3D<double> b = bias[c];
//std::cout << "FILTER " << c << std::endl;

			// Calculate gradients w.r.t. weights.
			// Convolve over padded input with the upstream gradient (dOut).
			int index = 1 + (padded.GetShape().height - output.GetShape().height) / stride;
			for (int h = 0; h < index; ++h) {
				for (int w = 0; w < index; ++w) {
					// Calculate the boundary indexes of the slice.
					vert_start = h * stride;
					vert_end = vert_start + out_shape.height;
					horiz_start = w * stride;
					horiz_end = horiz_start + out_shape.height;
//if (c >=3)					std::cout << "h: " << h << ", w: " << w << std::endl;// << ", h_s: " << horiz_start << ", h_e: " << horiz_end << std::endl;

					//dotProduct = 0;
					// Dot product between input-slice and dOut along the depth dimension.
					for (int weight_depth = 0; weight_depth < input.GetShape().depth; ++weight_depth) {
						dotProduct = 0;
						w_row = 0;
						for (int v_slice = vert_start; v_slice < vert_end; ++v_slice) {
							w_col = 0;
							for (int h_slice = horiz_start; h_slice < horiz_end; ++h_slice) {
//if (c >=3)								std::cout << "v_s: " << v_slice << ", h_s: "  << h_slice << ", w_d: " << weight_depth << std::endl;
//if (c >= 3)								std::cout << "w_r: " << w_row << ", w_c: " << w_col << std::endl;

								// Calculate grad_weights.
								// dW =  X * dOut
								inp = padded(v_slice, h_slice, weight_depth);
								double dO = grad_output(w_row, w_col, c);
								dotProduct += dO * inp;
//if (c >=3)								std::cout << "g.w.I: " << inp << ", W: " << dO << std::endl;

								++w_col;
							}
							++w_row;
						}
						grad_weights[c](h, w, weight_depth) = dotProduct;
					}
//					grad_weights[c](h, w, c) = dotProduct;
				}
			}
		}
	}

	std::vector<Tensor3D<double>>& Conv::GetWeights() {
		return weights;
	}

	std::vector<Tensor3D<double>>& Conv::GetBias() {
		return bias;
	}

	std::vector<Tensor3D<double>>& Conv::GetGradWeights() {
		return grad_weights;
	}

	std::vector<Tensor3D<double>>& Conv::GetGradBias() {
		return grad_bias;
	}

	Tensor3D<double> Conv::GetGradInput()
	{
		return grad_input;
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

	Tensor3D<double> Conv::Unpad(Tensor3D<double> padded) {
		convnet_core::Triplet input_shape = input.GetShape();
		convnet_core::Triplet padded_shape = padded.GetShape();
		Tensor3D<double> unpadded(input_shape.height, 
								  input_shape.width,
								  input_shape.depth);
		unpadded.InitZeros();

		for (int i = padding; i < padded_shape.height-padding; ++i) {
			for (int j = padding; j < padded_shape.width-padding; ++j) {
				for (int k = 0; k < padded_shape.depth; ++k) {
					unpadded(i-padding, j-padding, k) = padded(i, j, k);
				}
			}
		}

		return unpadded;
	}
}
