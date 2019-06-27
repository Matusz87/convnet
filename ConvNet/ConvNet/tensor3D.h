#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <random>

namespace convnet_core {
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
		Tensor3D<T> operator/(T scalar);
		Tensor3D<T> operator=(const Tensor3D<T>& other);
		
		T& operator()(int row, int col, int channel);
		T& get(int row, int col, int channel);

		Triplet GetShape();
		void InitZeros();
		void InitRandom();
		//void CopyFrom(std::vector<std::vector<std::vector<T>>> data);

		~Tensor3D();

	private:
		T* data;
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
		data = new T[other.shape.width *other.shape.height *other.shape.depth];
		memcpy(
			this->data,
			other.data,
			other.shape.width *other.shape.height *other.shape.depth * sizeof(T)
		);
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
	inline Tensor3D<T> Tensor3D<T>::operator/(T scalar) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] = this.data[i] / scalar;

		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator=(const Tensor3D<T>& other) {
		data = new T[other.shape.width *other.shape.height *other.shape.depth];
		memcpy(
			this->data,
			other.data,
			other.shape.width *other.shape.height *other.shape.depth * sizeof(T)
		);
		this->shape = other.shape;

		return *this;
	}
	
	/*template<typename T>
	void Tensor3D<T>::copy_from(std::vector<std::vector<std::vector<T>>> data) {
		int z = data.shape();
		int y = data[0].shape();
		int x = data[0][0].shape();

		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++)
				for (int k = 0; k < z; k++)
					get(i, j, k) = data[k][j][i];
	}*/

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
	inline Triplet Tensor3D<T>::GetShape() {
		return this->shape;
	}

	template<typename T>
	Tensor3D<T>::~Tensor3D() {
		if (data != NULL)
			delete[] data;
	}

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
		data = new T[height * width * depth];
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

//		double const distributionRangeHalfWidth = (2.4 / m_numInputs);
//		double const standardDeviation = distributionRangeHalfWidth * 2 / 6;

		// He initialization.
		double n = shape.height*shape.width*shape.depth;
		double const standard_dev = std::sqrt(2.0 / n);
		std::normal_distribution<> normalDistribution(0, standard_dev);
//		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
//		m.weight.data.normal_(0, math.sqrt(2. / n))

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
					printf("%.2f ", (double)tensor.get(x, y, z));
				}
				printf("\n");
			}
		}
	}

	static Tensor3D<float> to_tensor(std::vector<std::vector<std::vector<float>>> data) {
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();


		Tensor3D<float> t(x, y, z);

		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++)
				for (int k = 0; k < z; k++)
					t(i, j, k) = data[k][j][i];
		return t;
	}
}