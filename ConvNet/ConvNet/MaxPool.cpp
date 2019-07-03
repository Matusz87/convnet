#include "MaxPool.h"
#include <limits>

namespace layer {
	MaxPool::MaxPool() { }
	MaxPool::~MaxPool() { }

	MaxPool::MaxPool(std::string name, int height, int width, int depth,
					 int stride, int pool_size) : Layer(name) {
		LayerType type = LayerType::Pool;
		Layer::SetType(type);

		input = Tensor3D<double>(height, width, depth);
		int out_height = (height - pool_size) / stride + 1;
		int out_width = (width - pool_size) / stride + 1;
		output = Tensor3D<double>(out_height, out_width, depth);
		grad_input = Tensor3D<double>(height, width, depth);
		this->stride = stride;
		this->pool_size = pool_size;
	}
	MaxPool::MaxPool(convnet_core::Triplet shape, std::string name,
					 int stride, int pool_size) : Layer(shape, name) {
		int out_height = (shape.height - pool_size) / stride + 1;
		int out_width = (shape.width - pool_size) / stride + 1;
		output = Tensor3D<double>(out_height, out_width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input.InitZeros(); 

		this->stride = stride;
		this->pool_size = pool_size;
	}

	MaxPool::MaxPool(convnet_core::Tensor3D<double>& prev_activation, std::string name,
					 int stride, int pool_size) : Layer(prev_activation, name) {
		convnet_core::Triplet shape = prev_activation.GetShape();
		int out_height = (shape.height - pool_size) / stride + 1;
		int out_width = (shape.width - pool_size) / stride + 1;
		output = Tensor3D<double>(out_height, out_width, shape.depth);
		grad_input = Tensor3D<double>(shape.height, shape.width, shape.depth);
		grad_input.InitZeros();

		this->stride = stride;
		this->pool_size = pool_size;
	}

	void MaxPool::Forward(Tensor3D<double> prev_activation) {
//		std::cout << "Maxpool forward" << std::endl;
		input = Tensor3D<double>(prev_activation);
		max_indexes.clear();

		convnet_core::Triplet out_shape = GetOutputShape();

		int vert_start, vert_end, horiz_start, horiz_end;
		// Placeholder of maximum value in every slice.
		double max = std::numeric_limits<double>::lowest();
		convnet_core::Triplet max_index;
		double elem;
		for (int h = 0; h < out_shape.height; ++h) {
			for (int w = 0; w < out_shape.width; ++w) {
				for (int c = 0; c < out_shape.depth; ++c) {
					// Calculate the boundary indexes of the slice.
					vert_start = h * stride;
					vert_end = vert_start + pool_size;
					horiz_start = w * stride;
					horiz_end = horiz_start + pool_size;

					// Find the maximum value of the slice and save it.
					max = std::numeric_limits<double>::lowest();
					for (int v_slice = vert_start; v_slice < vert_end; ++v_slice) {
						for (int h_slice = horiz_start; h_slice < horiz_end; ++h_slice) {
							elem = input(v_slice, h_slice, c);
							if (elem > max) {
								max = elem;
								max_index.height = v_slice;
								max_index.width = h_slice;
								max_index.depth = c;
							}								
						}
					}
					output(h, w, c) = max;
//					std::cout << max_index.height << " " << max_index.width << " " << max_index.depth << std::endl;
					max_indexes.push_back(max_index);
				}
			}
		}
	}

	void MaxPool::Backprop(Tensor3D<double> grad_out) {
		grad_input.InitZeros();
		convnet_core::Triplet grad_shape = GetOutputShape();
		assert(grad_shape.height * grad_shape.width * grad_shape.depth == max_indexes.size());

		//for (auto ind : max_indexes) {
		//	std::cout << ind.height << " " << ind.width << " " << ind.depth << std::endl;
		//}

		int count = 0;
		for (int r = 0; r < grad_shape.height; ++r) {
			for (int c = 0; c < grad_shape.width; ++c) {
				for (int d = 0; d < grad_shape.depth; ++d) {
					convnet_core::Triplet index = max_indexes[count];
					double grad_val = grad_out(r, c, d);
					grad_input(index.height, index.width, index.depth) = grad_val;
					++count;
				}
			}
		}
	}

	void MaxPool::UpdateWeights(double lr, double momentum) { }

	nlohmann::json MaxPool::Serialize() {
		nlohmann::json layer;

		layer["type"] = "pool";
		layer["name"] = name;
		layer["height"] = GetInputShape().height;
		layer["width"] = GetInputShape().width;
		layer["depth"] = GetInputShape().depth;
		layer["p_size"] = GetPoolSize();
		layer["stride"] = GetStride();;
		
		return layer;
	}

	int MaxPool::GetPoolSize() {
		return pool_size;
	}

	int MaxPool::GetStride() {
		return stride;
	}
}