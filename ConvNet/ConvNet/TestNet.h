#pragma once

class TestNet
{
public:
	bool TestReluPool();
	bool TrainConvLayer();
	bool TrainFC();
	bool TrainFC2();
	bool TrainMNISTdigit();
	bool TrainSignFC();
	bool TrainSignCE();
	bool TrainSignCE2();
	bool TrainSignSimple();
	bool TrainSign();
	void TestJSON();
};
