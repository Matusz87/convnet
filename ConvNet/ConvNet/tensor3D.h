// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tamás Matuszka

#pragma once

#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <random>

namespace convnet_core {
	// Structure that stores the shape of a tensor.
	struct Triplet {
		int height, width, depth;
	};

	// Core data structure of the project. Stores the 3D volume of data in 
	// an unrolled vector. 
	template<typename T>
	class Tensor3D
	{
	public:
		Tensor3D() { };
		Tensor3D(Triplet shape);
		Tensor3D(int height, int width, int depth);
		Tensor3D(const Tensor3D& other);
		Tensor3D(const cv::Mat& image);
		Tensor3D<T> operator+(const Tensor3D<T>& other);
		Tensor3D<T> operator-(const Tensor3D<T>& other);
		Tensor3D<T> operator*(const Tensor3D<T>& other);
		Tensor3D<T> operator*(T scalar);
		Tensor3D<T> operator/(T scalar);
		Tensor3D<T> operator+(T scalar);
		Tensor3D<T> operator=(const Tensor3D<T>& other);

		T& operator()(int row, int col, int channel);
		T& get(int row, int col, int channel);
		// Sums a tensor, returns with a scalar.
		T Sum();
		// Applies the sign function to a tensor.
		Tensor3D<T> Sign();
		// Unrolls a 3D tensor into a (n, 1, 1) dimensional vector.
		Tensor3D<T> Flatten();
		// Reshapes a tensor to the given shape.
		Tensor3D<T> Reshape(Triplet shape);

		Triplet GetShape();
		void InitZeros();
		void InitRandom();

		~Tensor3D();

	private:
		// std vector stores elements, takes care of memory handling.
		std::vector<T> data;
		Triplet shape;

		// Appends an image channel (as a matrix) to the tensor using depth dimension.
		void AppendChannel(const cv::Mat& mat, int channel);
		void SetParams(int height, int width, int depth);
	};

	// Creates a tensor from a given shape.
	template<typename T>
	Tensor3D<T>::Tensor3D(Triplet shape) {
		SetParams(shape.height, shape.width, shape.depth);
	}

	// Creates a tensor from given dimensions.
	template<typename T>
	Tensor3D<T>::Tensor3D(int height, int width, int depth) {
		SetParams(height, width, depth);
	}

	// Copy constructor, creates a deep copy.
	template<typename T>
	Tensor3D<T>::Tensor3D(const Tensor3D& other) {
		data = std::vector<T>(other.data);
		this->shape = other.shape;
	}

	// Creates a tensor from an OpenCV image.
	template<typename T>
	Tensor3D<T>::Tensor3D(const cv::Mat& image) {
		int depth = image.channels();
		int height = image.rows;
		int width = image.cols;

		SetParams(height, width, depth);

		std::vector<cv::Mat> bgr(depth);
		cv::split(image, bgr);

		for (int i = 0; i < depth; ++i) {
			AppendChannel(bgr[i], i);
		}
	}
	
	// Adds two tensors (element-wise).
	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator+(const Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] += other.data[i];

		return clone;
	}

	// Subtracts a tensor from this tensor (element-wise).
	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator-(const Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] -= other.data[i];

		return clone;
	}

	// Multiplies two tensors (element-wise).
	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator*(const Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] *= other.data[i];

		return clone;
	}

	// Multiplies a tensor with a scalar.
	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator*(T scalar) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] *= scalar;

		return clone;
	}

	// Divides a tensor with a scalar.
	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator/(T scalar) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] = this->data[i] / scalar;

		return clone;
	}

	// Adds a scalar to a tensor.
	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator+(T scalar) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] = this->data[i] + scalar;

		return clone;
	}

	// Assignment operator.
	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator=(const Tensor3D<T>& other) {
		data = std::vector<T>(other.data);
		this->shape = other.shape;

		return *this;
	}

	// Indexing operator for the 3D tensor.
	template<typename T>
	inline T & Tensor3D<T>::operator()(int row, int col, int channel) {
		return this->get(row, col, channel);
	}

	// Auxiliary function that returns the data element from the unrolled vector.
	template<typename T>
	T& Tensor3D<T>::get(int row, int col, int channel) {
		assert(row >= 0 && col >= 0 && channel >= 0);
		assert(col < shape.width && row < shape.height && channel < shape.depth);

		return data[
			channel * (shape.width * shape.height) +
				row * (shape.width) +
				col
		];
	}

	template<typename T>
	inline T Tensor3D<T>::Sum() {
		return std::accumulate(data.begin(), data.end(), 0);;
	}

	template<typename T>
	Tensor3D<T> Tensor3D<T>::Sign() {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++) {
			if (clone.data[i] > 0)
				clone.data[i] = 1;
			if (clone.data[i] < 0)
				clone.data[i] = -1;
		}

		return clone;
	}

	// Creates a (height*width*depth, 1, 1) dimensional
	// copy of original tensor.
	template<typename T>
	Tensor3D<T> Tensor3D<T>::Flatten() {
		Tensor3D<T> clone(*this);
		int size = shape.width * shape.height * shape.depth;
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] = this->data[i];

		clone.shape.height = size;
		clone.shape.width = 1;
		clone.shape.depth = 1;

		return clone;
	}

	// Returns an (new_height, new_width, new_depth) dimensional
	// copy of the original tensor.
	template<typename T>
	Tensor3D<T> Tensor3D<T>::Reshape(Triplet new_shape) {
		Tensor3D<T> clone(*this);
		int size = shape.width * shape.height * shape.depth;
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] = this->data[i];

		clone.shape.height = new_shape.height;
		clone.shape.width = new_shape.width;
		clone.shape.depth = new_shape.depth;

		return clone;
	}

	template<typename T>
	inline Triplet Tensor3D<T>::GetShape() {
		return this->shape;
	}

	template<typename T>
	Tensor3D<T>::~Tensor3D() { }

	// Append the color-channels of an image to the unrolled data array.
	// @param mat: channel of an image as OpenCV matrix.
	// @param channel: number of channel.
	template<typename T>
	void Tensor3D<T>::AppendChannel(const cv::Mat& mat, int channel) {
		assert(channel >= 0);

		for (int i = 0; i < mat.rows; ++i) {
			for (int j = 0; j < mat.cols; ++j) {
				// Normalize pixel value between (0,1).
				get(i, j, channel) = (double)mat.at<uchar>(i, j) / 255.0;
			}
		}
	}

	template<typename T>
	void Tensor3D<T>::SetParams(int height, int width, int depth) {
		data = std::vector<T>(height * width * depth);

		shape.height = height;
		shape.width = width;
		shape.depth = depth;
	}

	// Sets all data elements to zero.
	template<typename T>
	void Tensor3D<T>::InitZeros() {
		for (int i = 0; i < shape.height; ++i)
			for (int j = 0; j < shape.width; ++j)
				for (int k = 0; k < shape.depth; ++k)
					get(i, j, k) = 0;
	}

	// Random initialization (used for CNN weights).
	// He initialization is used to ensure faster convergence.
	template<typename T>
	void Tensor3D<T>::InitRandom() {
		std::random_device rd;
		std::mt19937 generator(rd());

		// He initialization.
		double n = shape.height*shape.width*shape.depth;
		double const standard_dev = std::sqrt(2.0 / n);
		std::normal_distribution<> normalDistribution(0, standard_dev);

		for (int i = 0; i < shape.height; ++i)
			for (int j = 0; j < shape.width; ++j)
				for (int k = 0; k < shape.depth; ++k)
					get(i, j, k) = normalDistribution(generator);
	}

	// Utils functions.
	static void PrintTensor(Tensor3D<double>& tensor) {
		int width = tensor.GetShape().width;
		int height = tensor.GetShape().height;
		int depth = tensor.GetShape().depth;

		for (int z = 0; z < depth; z++) {
			printf("[Dim%d]\n", z);
			for (int x = 0; x < height; x++) {
				for (int y = 0; y < width; y++) {
					printf("%.4f ", (double)tensor.get(x, y, z));
				}
				printf("\n");
			}
		}
	}
}