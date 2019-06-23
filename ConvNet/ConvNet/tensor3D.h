#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <iostream>

namespace convnet_core {
	struct Shape {
		int height, width, depth;
	};

	template<typename T>
	class Tensor3D
	{
	public:
		Tensor3D(int height, int width, int depth);
		Tensor3D(const Tensor3D& other);
		Tensor3D(const cv::Mat& image);
		Tensor3D<T> operator+(Tensor3D<T>& other);
		Tensor3D<T> operator-(Tensor3D<T>& other);

		T& operator()(int _x, int _y, int _z);
		T& get(int _x, int _y, int _z);

		Shape GetShape();
		//void CopyFrom(std::vector<std::vector<std::vector<T>>> data);

		~Tensor3D();

	private:
		T * data;
		Shape shape;

		void AppendChannel(const cv::Mat& mat, int channel);
		void SetParams(int height, int width, int depth);
	};

	template<typename T>
	Tensor3D<T>::Tensor3D(int width, int height, int depth) {
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
	inline Tensor3D<T> Tensor3D<T>::operator+(Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] += other.data[i];
		
		return clone;
	}

	template<typename T>
	inline Tensor3D<T> Tensor3D<T>::operator-(Tensor3D<T>& other) {
		Tensor3D<T> clone(*this);
		for (int i = 0; i < other.shape.height * other.shape.width * other.shape.depth; i++)
			clone.data[i] -= other.data[i];

		return clone;
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
	inline Shape Tensor3D<T>::GetShape()
	{
		return this->shape;
	}

	template<typename T>
	Tensor3D<T>::~Tensor3D() {
		delete[] data;
	}

	// Append the color-channels of an image to the flattened data array.
	template<typename T>
	void Tensor3D<T>::AppendChannel(const cv::Mat& mat, int channel) {
		assert(channel >= 0);

		std::cout << "height: " << mat.rows << ", width: " << mat.cols << ", depth " << mat.channels() << std::endl;

		for (int i = 0; i < mat.rows; ++i) {
			for (int j = 0; j < mat.cols; ++j) {
				get(i, j, channel) = (int)mat.at<uchar>(i, j);
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

	static void PrintTensor(Tensor3D<float>& tensor) {
		int width = tensor.GetShape().width;
		int height = tensor.GetShape().height;
		int depth = tensor.GetShape().depth;

		for (int z = 0; z < depth; z++) {
			printf("[Dim%d]\n", z);
			for (int x = 0; x < height; x++) {
				for (int y = 0; y < width; y++) {
					printf("%.2f ", (float)tensor.get(x, y, z));
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
