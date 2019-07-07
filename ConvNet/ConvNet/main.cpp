#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "TestTensor.h"
#include "TestReLU.h"
#include "TestMaxPool.h"
#include "TestNet.h"
#include "TestConv.h"
#include "TestFC.h"
#include "TestSoftmax.h"

#include "MaxPool.h"
#include "Conv.h"
#include "ReLU.h"
#include "FC.h"
#include "Softmax.h"
#include "Model.h"
#include "Utils.h"

#include <iostream>
#include <string>

std::string dataset_path = "../../../datasets/traffic_signs/train-52x52/";

void Classify(std::string model_path, std::string image_path) {
	convnet_core::Model model;
	model.Load(model_path);

	cv::Mat img;
	img = cv::imread(image_path, cv::IMREAD_COLOR);
	cv::imshow(image_path, img);
	cv::waitKey(1);

	std::cout << "Classification result" << std::endl;
	convnet_core::PrintTensor(model.Predict(img));
}

void Evaluate(std::string model_path, std::string dataset_path) {
	convnet_core::Model model;
	model.Load(model_path);

	std::cout << "Loading test data..." << std::endl;
	utils::Dataset testSet = utils::GetTestSet(dataset_path, 500);
	Tensor3D<double> input, target, predicted;
	int correct = 0;
	
	for (int m = 0; m < testSet.size(); ++m) {
		if (m % 500 == 0) {
			std::cout << "... (" << m << "/" << testSet.size() << ")";
			std::cout << "Test set accuracy: " << 100* correct / (double)testSet.size() << "%" << std::endl;
		}

		input = testSet[m].first;
		target = testSet[m].second;

		predicted = model.Predict(input);

		if (utils::ComparePrediction(predicted, target))
			++correct;
	}

	std::cout << std::endl;
	std::cout << "Final Test set accuracy: " << 100 * correct / (double)testSet.size() << "%" << std::endl;
}

void Train(std::string model_path, std::string dataset_path, double lr, 
		   int epoch_num, int train_num, int valid_num, std::string model_name="cnn") {
	convnet_core::Model model;
	model.Load(model_path);

	std::cout << "Load training and validation sets..." << std::endl;
	utils::Dataset trainingSet = utils::GetTrainingSet(dataset_path, train_num);
	utils::Dataset validSet = utils::GetValidationSet(dataset_path, valid_num);
	Tensor3D<double> input, target;
	std::pair<bool, double> result;

	double cum_loss = 0;
	double cum_loss_valid = 0;
	int correct = 0;
	for (int epoch = 1; epoch <= epoch_num; ++epoch) {
		std::cout << "Epoch " << epoch << std::endl;
		std::srand(unsigned(std::time(0)));
		std::random_shuffle(trainingSet.begin(), trainingSet.end());

		cum_loss = 0;
		cum_loss_valid = 0;
		correct = 0;
		for (int m = 0; m < trainingSet.size(); ++m) {
			if (m % 500 == 0) {
				std::cout << "... (" << m << "/" << trainingSet.size() << ")";
			}

			input = trainingSet[m].first;
			target = trainingSet[m].second;

			result = model.Fit(input, target, lr);

			if (result.first)
				++correct;

			cum_loss += result.second;
		}

		if (epoch % 1 == 0) {
			model.Save("models/model-checkpoint-" + std::to_string(epoch) + ".json");
		}
		if (epoch == epoch_num)
			model.Save("models/model-" + model_name + std::to_string(epoch) + ".json");

		std::cout << std::endl;
		std::cout << "Epoch " << epoch << " is done. " << std::endl;
		std::cout << "Training loss: " << cum_loss / (double)trainingSet.size() << std::endl;
		std::cout << "Training accuracy: " << 100 * correct / (double)trainingSet.size() << "%" << std::endl;

		// Calculate validation loss
		correct = 0;
		for (int m = 0; m < validSet.size(); ++m) {
			input = validSet[m].first;
			target = validSet[m].second;

			result = model.Evaluate(input, target);
			cum_loss_valid += result.second;

			if (result.first)
				++correct;

		}

		std::cout << "Validation loss: " << cum_loss_valid / (double)validSet.size() << std::endl;
		std::cout << "Validation accuracy: " << 100 * correct / (double)validSet.size() << "%" << std::endl << std::endl;
	}
}

int main(int argc, char** argv) {
	if (argc == 3) {
		Evaluate(argv[1], argv[2]);
	} else if (argc == 4) {
		Classify(argv[2], argv[3]);
	}
	else if (argc == 8) {
		Train(argv[1], argv[2], atof(argv[3]), atoi(argv[4]), 
			  atoi(argv[5]), atoi(argv[6]), argv[7]);
	} else {
		std::cout << "Usage:" << std::endl;
		std::cout << "Classifiy one image: ConvNet.exe -c model_path image_path" << std::endl;
		std::cout << "Evaluate model on test set: ConvNet.exe model_path dataset_path" << std::endl;
		std::cout << "Train model: ConvNet.exe model_path dataset_path learning_rate "
			<< "epochs train_set_size valid_set_size model_name" << std::endl;
	}

	return 0;
}