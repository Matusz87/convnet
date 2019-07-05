// PROJECT: Convolutional neural network implementation.
// AUTHOR: Tam�s Matuszka

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
		T Sum();
		Tensor3D<T> Sign();
		Tensor3D<T> Flatten();
		Tensor3D<T> Reshape(Triplet shape);

		Triplet GetShape();
		void InitZeros();
		void InitRandom();

		~Tensor3D();

	private:
		std::vector<T> data;
		Triplet shape;

		void AppendChannel(const cv::Mat& mat, int channel);
		void SetParams(int height, int width, int depth);
	};

	template<typename T>
	Tensor3D<T>::Tensor3D(Triplet shape) {
		SetParams(shape.height, shape.width, shape.depth);
	}

	template<typename T>
	Tensor3D<T>::Tensor3D(int height, int width, int depth) {
		SetParams(height, width, depth);
	}

	template<typename T>
	Tensor3D<T>::Tensor3D(const Tensor3D& other) {
		data = std::vector<T>(other.data);
		this->shape = other.shape;
	}

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

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator+(const Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] += other.data[i];

		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator-(const Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] -= other.data[i];

		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator*(const Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] *= other.data[i];

		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator*(T scalar) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] *= scalar;

		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator/(T scalar) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] = this->data[i] / scalar;

		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator+(T scalar) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < shape.height * shape.width * shape.depth; i++)
			clone.data[i] = this->data[i] + scalar;

		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator=(const Tensor3D<T>& other) {
		data = std::vector<T>(other.data);
		this->shape = other.shape;

		return *this;
	}

	template<typename T>
	inline T & Tensor3D<T>::operator()(int row, int col, int channel) {
		return this->get(row, col, channel);
	}

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

	template<typename T>
	void Tensor3D<T>::InitZeros() {
		for (int i = 0; i < shape.height; ++i)
			for (int j = 0; j < shape.width; ++j)
				for (int k = 0; k < shape.depth; ++k)
					get(i, j, k) = 0;
	}

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