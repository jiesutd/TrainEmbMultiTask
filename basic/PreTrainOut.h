/*
 * DenseFeatureChar.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef PRETRAINOUT_H_
#define PRETRAINOUT_H_

#include "N3L.h"

template<typename xpu>
class PreTrainOut {
public:
	//all state inter dependent features

	Tensor<xpu, 3, dtype> _charOut;
	Tensor<xpu, 3, dtype> _charOut_Loss;
	bool _bTrain;
	int _charnum;


public:
	PreTrainOut() {
		_bTrain = false;
		_charnum = 0;
	}

	~PreTrainOut() {
		clear();
	}

public:
	inline void init(int charnum, int charOutSize, bool bTrain = false) {
		clear();
		_charnum = charnum;
		_bTrain = bTrain;
		_charOut = NewTensor<xpu>(Shape3(_charnum, charOutSize), d_zero);

		if (_bTrain) {
			_charOut_Loss = NewTensor<xpu>(Shape3(_charnum, charOutSize), d_zero);
		}
	}

	inline void clear() {
		FreeSpace(&_charOut);
		if (_bTrain) {
			FreeSpace(&_charOut_Loss);
		}
		_bTrain = false;
		_charnum = 0;
	}

};

#endif /* PRETRAINOUT_H_ */
