#pragma once
class TestConv
{
public:
	TestConv();
	~TestConv();
	bool TestConstructor();
	bool TestPadding();
	bool TestUnpadding();
	bool TestForward();
	bool TestForwardPadded();
	bool TestForward2();
	bool TestForwardDeep();
	bool TestForward3();
	bool TestForwardWithMat();
	bool TestBackprop();
	bool TestBackpropPadded();
};

