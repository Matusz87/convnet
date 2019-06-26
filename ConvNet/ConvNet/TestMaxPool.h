#pragma once
class TestMaxPool
{
public:
	bool TestConstructor();
	bool TestConstructorWithTensor();
	bool TestConstructorWithMat();
	bool TestForward();
	bool TestForwardWithMat();
	bool TestBackprop();
};

