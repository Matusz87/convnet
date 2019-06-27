#pragma once
class TestConv
{
public:
	TestConv();
	~TestConv();
	bool TestConstructor();
	bool TestPadding();
	bool TestForward();
	bool TestForwardPadded();
	bool TestForward2();
	bool TestForwardDeep();
	bool TestForward3();
	bool TestForwardWithMat();
	bool TestBackprop();
};

