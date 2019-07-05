// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#include "Conv.h"

namespace layer {
	// Default constructor and destructor.
	Conv::Conv() { }
	Conv::~Conv() { }

	// Creates a Conv layers invoking the base constructor.
	// @param height:	input height
	// @param width:	input width
	// @param depth:	input depth
	// @param name:		name of the layer
	// @param f_count:	number of filters in the layer
	// @param f_size:	filter size (filter shape is (f_size, f_size, depth)
	// @param stride:	step size during sliding.
	// @param paddig:	how many zeros will be added around tensor.
	Conv::Conv(int height, int width, int depth, std::string name, int f_count,
		int f_size, int stride, int padding) : Layer(name) {
		LayerType type = LayerType::Conv;
		Layer::SetType(type);
		input = Tensor3D<double>(height, width, depth);
		filter_count = f_count;
		filter_size = f_size;
		this->stride = stride;
		this->padding = padding;

		int out_height = ((height - f_size + 2 * padding) / stride) + 1;
		int out_width = ((width - f_size + 2 * padding) / stride) + 1;
		output = Tensor3D<double>(out_height, out_width, f_count);

		this->name = name;

		// Init variables.
		InitWeights();
		InitBias();
		InitGrads();
	}

	// Copy constructor, used for loading parameters from a saved model.
	// @param other: layer which will be copied.
	Conv::Conv(const Conv& other) : Layer(other) {
		weights = other.weights;
		bias = other.bias;
		InitGrads();
	}

	// Creates a Conv layers invoking the base constructor.
	// @param shape:	input dimensions
	// @param name:		name of the layer
	// @param f_count:	number of filters in the layer
	// @param f_size:	filter size (filter shape is (f_size, f_size, depth)
	// @param stride:	step size during sliding.
	// @param paddig:	how many zeros will be added around tensor.
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

	// Creates a Conv layers invoking the base constructor.
	// @param prev_act:	tensor from previous layer
	// @param name:		name of the layer
	// @param f_count:	number of filters in the layer
	// @param f_size:	filter size (filter shape is (f_size, f_size, depth)
	// @param stride:	step size during sliding.
	// @param paddig:	how many zeros will be added around tensor.
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

	// Forward pass. Performs convolutions of weights with the input volume.
	// @param prev_act:	activation map from previous layer
	void Conv::Forward(const Tensor3D<double>& prev_activation) {
		input = Tensor3D<double>(prev_activation);

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

	// Calculates the gradients based on the upstream gradient.
	// Can be interpreted as a convolution.
	void Conv::Backprop(Tensor3D<double>& grad_output) {
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
//					std::cout << "h: " << h << ", w: " << w << std::endl;// << ", h_s: " << horiz_start << ", h_e: " << horiz_end << std::endl;

					dotProduct = 0;
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
//								std::cout << "g.i.I: " << inp << ", W: " << dO << std::endl << std::endl;

								++w_col;
							}
							++w_row;
						}
					}
				}
			}
			grad_bias[c](0, 0, 0) = sum_dOut;
		}
		// Handle zero-padding.
		grad_input = Tensor3D<double>(Unpad(grad_input_padded));

		for (int c = 0; c < out_shape.depth; ++c) {
			Tensor3D<double> W = weights[c];
			Tensor3D<double> b = bias[c];

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
				}
			}
		}
	}

	// Adjudsts weights based on the calculated gradients. 
	// Uses Nesterov Accelerated Gradient method.
	// @param lr:		learning rate
	// @param momentum: momentum
	// Further description of method: http://cs231n.github.io/neural-networks-3/#sgd
	void Conv::UpdateWeights(double lr, double momentum) {
		for (int i = 0; i < grad_weights.size(); ++i) {
			Tensor3D<double> v_prev(velocities[i]);
			velocities[i] = velocities[i] *momentum - (grad_weights[i] *lr);
			weights[i] = weights[i] - (v_prev*momentum) + v_prev*(1 + momentum);
		}
		
		for (int i = 0; i < grad_bias.size(); ++i) {
			bias[i] = bias[i] - (bias[i]*lr);
		}
	}

	// Store layer parameters in a JSON node.
	// returns layer: JSON representation of the layer. 
	nlohmann::json Conv::Serialize() {
		nlohmann::json layer;
		nlohmann::json weights_json;
		nlohmann::json bias_json;

		layer["type"] = "conv";
		layer["name"] = name;
		layer["height"] = GetInputShape().height;
		layer["width"] = GetInputShape().width;
		layer["depth"] = GetInputShape().depth;
		layer["f_count"] = filter_count;
		layer["f_size"] = filter_size;
		layer["stride"] = stride;;
		layer["padding"] = padding;

		for (int filter = 0; filter < weights.size(); ++filter) {
			nlohmann::json weight;
			for (int d = 0; d < weights[filter].GetShape().depth; ++d)
				for (int h = 0; h < weights[filter].GetShape().height; ++h)
					for (int w = 0; w < weights[filter].GetShape().width; ++w)
						weight.push_back(weights[filter](h, w, d));

			weights_json[std::to_string(filter)] = weight;
			bias_json.push_back(bias[filter](0, 0, 0));
		}
		layer["weights"] = weights_json;
		layer["bias"] = bias_json;

		return layer;
	}

	// Not needed.
	double Conv::Loss(Tensor3D<double>& target) { return 0.0; }

	// Getter methods.
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

	// He initialization.
	void Conv::InitWeights() {
		weights = std::vector<Tensor3D<double>>(filter_count);
		for (int i = 0; i < filter_count; ++i) {
			Tensor3D<double> t(filter_size, filter_size, input.GetShape().depth);
			t.InitRandom();
			weights[i] = t;			
		}
	}

	void Conv::InitBias() {
		bias = std::vector<Tensor3D<double>>(filter_count);
		for (int i = 0; i < filter_count; ++i) {
			Tensor3D<double> t(1, 1, 1);
			t.InitZeros();
			bias[i] = t;
		}
	}

	void Conv::InitGrads() {
		grad_input = Tensor3D<double>(input.GetShape());
		grad_input.InitZeros();

		grad_weights = std::vector<Tensor3D<double>>(filter_count);
		grad_bias = std::vector<Tensor3D<double>>(filter_count);
		velocities = std::vector<Tensor3D<double>>(filter_count);

		for (int i = 0; i < filter_count; ++i) {
			Tensor3D<double> dB(1, 1, 1);
			dB.InitZeros();
			grad_bias[i] = dB;

			Tensor3D<double> dW(filter_size, filter_size, input.GetShape().depth);
			dW.InitZeros();
			grad_weights[i] = dW;
			velocities[i] = Tensor3D<double>(dW);
		}
	}

	// Add zeros around each matrices along depth dimension.
	// param tensor:	tensor on which padding will be applied
	// returns padded:	zero-padded tensor
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

	// Getter methods for deserializing.
	int Conv::GetFilterCount() {
		return filter_count;
	}

	int Conv::GetFilterSize() {
		return filter_size;
	}

	int Conv::GetStride() {
		return stride;
	}

	int Conv::GetPadding() {
		return padding;
	}

	// Removes zero-padding from a tensor.
	// param padded: zero padded tensor
	// return unpadded: unpadded tensor
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
