/*
 * DenseFeatureChar.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef FEATURE_DENSEFEATURECHARWITHMAP_H_
#define FEATURE_DENSEFEATURECHARWITHMAP_H_

#include "N3L.h"

template<typename xpu>
class DenseFeatureCharWithMap {
public:
	//all state inter dependent features
	vector<int> _charIds, _bicharIds;
	Tensor<xpu, 3, dtype> _charprime, _bicharprime,_mapcharprime, _mapcharHidden;
	Tensor<xpu, 3, dtype> _charpre, _charpreMask;
	Tensor<xpu, 3, dtype> _charInput, _charHidden;
	Tensor<xpu, 2, dtype> _charDummy;

	Tensor<xpu, 3, dtype> _charprime_Loss, _bicharprime_Loss,_mapcharprime_Loss, _mapcharHidden_Loss, _charpre_Loss;
	Tensor<xpu, 3, dtype> _charInput_Loss, _charHidden_Loss;
	Tensor<xpu, 2, dtype> _charDummy_Loss;

	Tensor<xpu, 3, dtype>  _charHidden2, _charHidden2_Loss;
	Tensor<xpu, 3, dtype> _charOut, _charOut_Loss;

	bool _bTrain;
	int _charnum;
	int _buffer;

public:
	DenseFeatureCharWithMap() {
		_bTrain = false;
		_charnum = 0;
		_buffer = 0;
	}

	~DenseFeatureCharWithMap() {
		clear();
	}

public:
	inline void init(int charnum, int charDim, int bicharDim, int mapcharDim, int mapHiddenSize, int charcontext, int charHiddenDim, int charOutSize, int buffer = 0, bool bTrain = false) {
		clear();

		_charnum = charnum;
		_bTrain = bTrain;
		_buffer = buffer;

		if (_charnum > 0) {
			int charwindow = 2 * charcontext + 1;
			int charpresize = charDim + bicharDim + mapHiddenSize;
			int charRepresentDim = charpresize * charwindow;

			_charIds.resize(charnum);
			_bicharIds.resize(charnum);
			_charprime = NewTensor<xpu>(Shape3(_charnum, 1, charDim), d_zero);
			_bicharprime = NewTensor<xpu>(Shape3(_charnum, 1, bicharDim), d_zero);
			_mapcharprime = NewTensor<xpu>(Shape3(_charnum, 1, mapcharDim), d_zero);
			_mapcharHidden = NewTensor<xpu>(Shape3(_charnum, 1, mapHiddenSize), d_zero);
			_charpre = NewTensor<xpu>(Shape3(_charnum, 1, charpresize), d_zero);

			_charInput = NewTensor<xpu>(Shape3(_charnum, 1, charRepresentDim), d_zero);
			_charHidden = NewTensor<xpu>(Shape3(_charnum, 1, charHiddenDim), d_zero);
			_charHidden2 = NewTensor<xpu>(Shape3(_charnum, 1, charHiddenDim), d_zero);
			_charDummy = NewTensor<xpu>(Shape2(1, charHiddenDim), d_zero);

			_charOut = NewTensor<xpu>(Shape3(_charnum, 1, charOutSize), d_zero);

			if (_bTrain) {
				_charpreMask = NewTensor<xpu>(Shape3(_charnum, 1, charpresize), d_zero);
				_charprime_Loss = NewTensor<xpu>(Shape3(_charnum, 1, charDim), d_zero);
				_bicharprime_Loss = NewTensor<xpu>(Shape3(_charnum, 1, bicharDim), d_zero);
				_mapcharprime_Loss = NewTensor<xpu>(Shape3(_charnum, 1, mapcharDim), d_zero);
				_mapcharHidden_Loss = NewTensor<xpu>(Shape3(_charnum, 1, mapHiddenSize), d_zero);
				_charpre_Loss = NewTensor<xpu>(Shape3(_charnum, 1, charpresize), d_zero);
				_charInput_Loss = NewTensor<xpu>(Shape3(_charnum, 1, charRepresentDim), d_zero);
				_charHidden_Loss = NewTensor<xpu>(Shape3(_charnum, 1, charHiddenDim), d_zero);
				_charHidden2_Loss = NewTensor<xpu>(Shape3(_charnum, 1, charHiddenDim), d_zero);
				_charDummy_Loss = NewTensor<xpu>(Shape2(1, charHiddenDim), d_zero);
				_charOut_Loss = NewTensor<xpu>(Shape3(_charnum, 1, charOutSize), d_zero);
			}
		}

	}

	inline void clear() {
		if (_charnum > 0) {
			_charIds.clear();
			_bicharIds.clear();

			FreeSpace(&_charprime);
			FreeSpace(&_bicharprime);
			FreeSpace(&_mapcharprime);
			FreeSpace(&_mapcharHidden);
			FreeSpace(&_charpre);
			FreeSpace(&_charInput);
			FreeSpace(&_charHidden);
			FreeSpace(&_charHidden2);
			FreeSpace(&_charDummy);
			FreeSpace(&_charOut);
			if (_bTrain) {
				FreeSpace(&_charprime_Loss);
				FreeSpace(&_bicharprime_Loss);
				FreeSpace(&_mapcharprime_Loss);
				FreeSpace(&_mapcharHidden_Loss);
				FreeSpace(&_charpreMask);
				FreeSpace(&_charpre_Loss);
				FreeSpace(&_charInput_Loss);
				FreeSpace(&_charHidden_Loss);
				FreeSpace(&_charHidden2_Loss);
				FreeSpace(&_charDummy_Loss);
				FreeSpace(&_charOut_Loss);
			}

		}

		_bTrain = false;
		_charnum = 0;
		_buffer = 0;
	}

};

#endif /* FEATURE_DENSEFEATURECHAR_H_ */
