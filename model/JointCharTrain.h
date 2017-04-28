/*
 * JointCharTrain.h
 *
 *  Created on: Dec 24, 2016
 *      Author: Jie
 *      Revised from CharStackWord.h
 *      Original from meishan zhang
 */

#ifndef SRC_TLSTMBeamSearcher_H_
#define SRC_TLSTMBeamSearcher_H_

#if defined __GNUC__ || defined __APPLE__
#include <ext/hash_set>
#else
#include <hash_set>
#endif

#include <iostream>
#include <assert.h>
#include "DenseFeatureCharWithMap.h"
#include "N3L.h"
#include "SegLookupTable.h"
#include "NewConcat.h"
#include "Example.h"
#include "NewSoftMaxLoss.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//re-implementation of Yue and Clark ACL (2007)
template<typename xpu>
class Classifier {
public:
	Classifier() {
		_dropOut = 0.5;
		_oovRatio = 0.2;
		_oovFreq = 3;
		_buffer = 0;
	}
	~Classifier() {
	}

public:
	//SparseUniLayer1O<xpu> _splayer_output;
	UniLayer<xpu> _nnlayer_char_hidden;
	UniLayer<xpu> _nnlayer_char_hidden2;
	UniLayer<xpu> _A_olayer_chartrain;
	UniLayer<xpu> _B_olayer_chartrain;
	UniLayer<xpu> _C_olayer_chartrain;
	UniLayer<xpu> _D_olayer_chartrain;
	UniLayer<xpu> _E_olayer_chartrain;
	int _A_labelSize = 0;
	int _B_labelSize = 0;
	int _C_labelSize = 0;
	int _D_labelSize = 0;
	int _E_labelSize = 0;
	LookupTable<xpu> _chars;
	LookupTable<xpu> _bichars;


	int _charSize = 0;
	int _biCharSize = 0;
	int _charDim = 0;
	int _biCharDim = 0;


	int _charcontext = 0;
	int _charwindow = 0;
	int _charRepresentDim = 0;

	int _charHiddenSize = 0;

	Metric _A_eval;
	Metric _B_eval;
	Metric _C_eval;
	Metric _D_eval;
	Metric _E_eval;

	dtype _dropOut;

	dtype _oovRatio;
	int _buffer;

	int _oovFreq = 0;

	int MAX_SENTENCE_SIZE = 512;

public:

	inline void init(const NRMat<dtype>& charEmb, const NRMat<dtype>& bicharEmb, int charcontext, int charHiddenSize,
	 int A_labelSize, int B_labelSize, int C_labelSize, int D_labelSize, int E_labelSize) {

		_charSize = charEmb.nrows();	
		_biCharSize = bicharEmb.nrows();
		
		_charDim = charEmb.ncols();
		_biCharDim = bicharEmb.ncols();
		_charHiddenSize = charHiddenSize;

		
		_charcontext = charcontext;
		_charwindow = 2 * charcontext + 1;
		_charRepresentDim = (_charDim + _biCharDim) * _charwindow;

		_chars.initial(charEmb);
		_chars.setEmbFineTune(true);
		_bichars.initial(bicharEmb);
		_bichars.setEmbFineTune(true);
		
		_nnlayer_char_hidden.initial(_charHiddenSize, _charRepresentDim, true, 100, 0);
		_nnlayer_char_hidden2.initial(_charHiddenSize, _charHiddenSize, true, 100, 0);
		
		_A_labelSize = A_labelSize;
		_B_labelSize = B_labelSize;
		_C_labelSize = C_labelSize;
		_D_labelSize = D_labelSize;
		_E_labelSize = E_labelSize;
		_A_olayer_chartrain.initial(_A_labelSize, _charHiddenSize, true, 60, 2);
		_B_olayer_chartrain.initial(_B_labelSize, _charHiddenSize, true, 60, 2);
		_C_olayer_chartrain.initial(_C_labelSize, _charHiddenSize, true, 60, 2);
		_D_olayer_chartrain.initial(_D_labelSize, _charHiddenSize, true, 60, 2);
		_E_olayer_chartrain.initial(_E_labelSize, _charHiddenSize, true, 60, 2);
		showParameters();
	}

	inline void release() {
		//_splayer_output.release();
		_chars.release();
		_bichars.release();
		_nnlayer_char_hidden.release();
		_nnlayer_char_hidden2.release();
		_A_olayer_chartrain.release();
		_B_olayer_chartrain.release();
		_C_olayer_chartrain.release();
		_D_olayer_chartrain.release();
		_E_olayer_chartrain.release();
	}

	dtype train(const vector<Example>& examples, const vector<string>& exampleTypes) {
		assert(examples.size()== exampleTypes.size());
		_A_eval.reset();
		_B_eval.reset();
		_C_eval.reset();
		_D_eval.reset();
		_E_eval.reset();
		dtype cost = 0.0;
		int batch_size = examples.size();
		for (int idx = 0; idx < examples.size(); idx++) {
			cost += trainOneExample(examples[idx],exampleTypes[idx]);
		}
		return cost;
	}


	dtype trainOneExample(const Example& example,const string& exampleType) {	
		srand(1);
		int cut_size = 4;
		int length = example.charIds.size();
		if (length >= MAX_SENTENCE_SIZE) {
			cout << "sentence size is "<<length<< " larger than " << MAX_SENTENCE_SIZE <<endl;
			return 0.0;
		}
		if (length == 0) {
			return 0.0;
		}
		dtype cost = 0.0;
		static DenseFeatureCharWithMap<xpu> charFeat;
		if (exampleType == "A") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize, _A_labelSize, _buffer, true);
		} else if (exampleType == "B") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize, _B_labelSize, _buffer, true);
		} else if (exampleType == "C") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize, _C_labelSize, _buffer, true);
		} else if (exampleType == "D") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize, _D_labelSize, _buffer, true);
		}else if (exampleType == "E") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize, _E_labelSize, _buffer, true);
		} else {
			cout << "trainoneexample error!"<<endl;
		}
		
		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = example.charIds[idx];
			charFeat._bicharIds[idx] = example.bicharIds[idx];
		}
		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);
			if ((exampleType == "B")|| (exampleType == "C")) {
				dropoutcol(charFeat._charpreMask[idx], 0.0);
			} else {
				dropoutcol(charFeat._charpreMask[idx], _dropOut);
			}
			
			charFeat._charpre[idx] = charFeat._charpre[idx] * charFeat._charpreMask[idx];
		}
		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);
		if (exampleType == "B") {
			for (int b_iter = cut_size; b_iter < length - cut_size; ++b_iter) {
				_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput[b_iter], charFeat._charHidden[b_iter]);
				_nnlayer_char_hidden2.ComputeForwardScore(charFeat._charHidden[b_iter], charFeat._charHidden2[b_iter]);
				_B_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2[b_iter], charFeat._charOut[b_iter]);
			}
			cost += newsoftmax_loss(charFeat._charOut, example.labels, charFeat._charOut_Loss, _B_eval, 1, cut_size);
			for (int b_iter = cut_size; b_iter < length - cut_size; ++b_iter) {
				_B_olayer_chartrain.ComputeBackwardLoss(charFeat._charHidden2[b_iter], charFeat._charOut[b_iter], charFeat._charOut_Loss[b_iter], charFeat._charHidden2_Loss[b_iter]);
				_nnlayer_char_hidden2.ComputeBackwardLoss(charFeat._charHidden[b_iter], charFeat._charHidden2[b_iter], charFeat._charHidden2_Loss[b_iter], charFeat._charHidden_Loss[b_iter]);
				_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput[b_iter], charFeat._charHidden[b_iter], charFeat._charHidden_Loss[b_iter], charFeat._charInput_Loss[b_iter]);
			}
		} else {
			_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
			_nnlayer_char_hidden2.ComputeForwardScore(charFeat._charHidden, charFeat._charHidden2);
			if (exampleType == "A") {
				_A_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
				cost += newsoftmax_loss(charFeat._charOut, example.labels, charFeat._charOut_Loss, _A_eval, 1);
				_A_olayer_chartrain.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charOut, charFeat._charOut_Loss, charFeat._charHidden2_Loss);
			} else if (exampleType == "C") {
				_C_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
				cost += newsoftmax_loss(charFeat._charOut, example.labels, charFeat._charOut_Loss, _C_eval, 1);
				_C_olayer_chartrain.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charOut, charFeat._charOut_Loss, charFeat._charHidden2_Loss);
			} else if (exampleType == "D") {
				_D_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
				cost += newsoftmax_loss(charFeat._charOut, example.labels, charFeat._charOut_Loss, _D_eval, 1);
				_D_olayer_chartrain.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charOut, charFeat._charOut_Loss, charFeat._charHidden2_Loss);
			} else if (exampleType == "E") {
				_E_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
				cost += newsoftmax_loss(charFeat._charOut, example.labels, charFeat._charOut_Loss, _E_eval, 1);
				_E_olayer_chartrain.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charOut, charFeat._charOut_Loss, charFeat._charHidden2_Loss);
			} else {
				cout << "trainoneexample error!"<<endl;
			}
			_nnlayer_char_hidden2.ComputeBackwardLoss(charFeat._charHidden, charFeat._charHidden2, charFeat._charHidden2_Loss, charFeat._charHidden_Loss);
			_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput, charFeat._charHidden, charFeat._charHidden_Loss, charFeat._charInput_Loss);
		}
		windowlized_backward(charFeat._charpre_Loss, charFeat._charInput_Loss, _charcontext);
		charFeat._charpre_Loss = charFeat._charpre_Loss * charFeat._charpreMask;
		for(int idx = 0; idx < length; idx++){
			unconcat(charFeat._charprime_Loss[idx], charFeat._bicharprime_Loss[idx], charFeat._charpre_Loss[idx]);
			_chars.EmbLoss(charFeat._charIds[idx], charFeat._charprime_Loss[idx]);
			_bichars.EmbLoss(charFeat._bicharIds[idx], charFeat._bicharprime_Loss[idx]);
		}

		charFeat.clear();
		if ((_A_eval.getAccuracy() < 0)||(_B_eval.getAccuracy() < 0)||(_C_eval.getAccuracy() < 0)||(_D_eval.getAccuracy() < 0)||(_E_eval.getAccuracy() < 0)) {
	      std::cout << "strange" << std::endl;
	    }
		return cost;
	}

	
	bool new_decode(const Example& example, vector<int>& results, const string& exampleType) {
		if (exampleType == "A") {
			_A_eval.reset();
		} else if (exampleType == "B") {
			_B_eval.reset();
		} else if (exampleType == "C") {
			_C_eval.reset();
		} else if (exampleType == "D") {
			_D_eval.reset();
		} else if (exampleType == "E") {
			_E_eval.reset();
		} else {
			cout << "new decode error!"<<endl;
		}
		
		dtype cost = 0.0;
		int length = example.charIds.size();
		if (length >= MAX_SENTENCE_SIZE) {
			cout << "sentence size is "<<length<< " larger than " << MAX_SENTENCE_SIZE <<endl;
			return 0.0;
		}
		static DenseFeatureCharWithMap<xpu> charFeat;
		if (exampleType == "A") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_A_labelSize, _buffer, true);
		} else if (exampleType == "B") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_B_labelSize, _buffer, true);
		} else if (exampleType == "C") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_C_labelSize, _buffer, true);
		} else if (exampleType == "D") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_D_labelSize, _buffer, true);
		} else if (exampleType == "E") {
			charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_E_labelSize, _buffer, true);
		} else {
			cout << "new decode error!"<<endl;
		}

		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = example.charIds[idx];
			charFeat._bicharIds[idx] = example.bicharIds[idx];
		}

		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);
		}
		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);
		_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
		_nnlayer_char_hidden2.ComputeForwardScore(charFeat._charHidden, charFeat._charHidden2);
		if (exampleType == "A") {
			_A_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
		} else if (exampleType == "B") {
			_B_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
		} else if (exampleType == "C") {
			_C_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
		} else if (exampleType == "D") {
			_D_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
		} else if (exampleType == "E") {
			_E_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
		} else {
			cout << "new decode error!"<<endl;
		}
		newsoftmax_predict(charFeat._charOut, results);
		charFeat.clear();
		return true;
	}

	// bool B_decode(const Example& example, vector<int>& results) {
	// 	_B_eval.reset();
	// 	dtype cost = 0.0;
	// 	int length = example.charIds.size();
	// 	if (length >= MAX_SENTENCE_SIZE) {
	// 		cout << "sentence size is "<<length<< " larger than " << MAX_SENTENCE_SIZE <<endl;
	// 		return 0.0;
	// 	}
	// 	static DenseFeatureCharWithMap<xpu> charFeat;

	// 	charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_B_labelSize, _buffer, true);
	// 	for (int idx = 0; idx < length; idx++) {
	// 		charFeat._charIds[idx] = example.charIds[idx];
	// 		charFeat._bicharIds[idx] = example.bicharIds[idx];
	// 	}

	// 	for (int idx = 0; idx < length; idx++) {
	// 		_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
	// 		_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
	// 		concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);
	// 	}
	// 	windowlized(charFeat._charpre, charFeat._charInput, _charcontext);
	// 	_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
	// 	_nnlayer_char_hidden2.ComputeForwardScore(charFeat._charHidden, charFeat._charHidden2);
	// 	_B_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);
	// 	newsoftmax_predict(charFeat._charOut, results);
	// 	charFeat.clear();
	// 	return true;
	// }


	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps, dtype clip = -1.0) {
		if(clip > 0.0) {
			dtype norm = 0.0;
			norm += _chars.squarenormAll();
			norm += _bichars.squarenormAll();

			norm += _nnlayer_char_hidden.squarenormAll();
			norm += _nnlayer_char_hidden2.squarenormAll();

			norm += _A_olayer_chartrain.squarenormAll();
			norm += _B_olayer_chartrain.squarenormAll();
			norm += _C_olayer_chartrain.squarenormAll();
			norm += _D_olayer_chartrain.squarenormAll();
			norm += _E_olayer_chartrain.squarenormAll();
			if(norm > clip * clip){
				dtype scale = clip/sqrt(norm);
				_chars.scaleGrad(scale);
				_bichars.scaleGrad(scale);
				_nnlayer_char_hidden.scaleGrad(scale);
				_nnlayer_char_hidden2.scaleGrad(scale);
				_A_olayer_chartrain.scaleGrad(scale);
				_B_olayer_chartrain.scaleGrad(scale);
				_C_olayer_chartrain.scaleGrad(scale);
				_D_olayer_chartrain.scaleGrad(scale);
				_E_olayer_chartrain.scaleGrad(scale);
			}
		}
		_chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_bichars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_A_olayer_chartrain.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_B_olayer_chartrain.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_C_olayer_chartrain.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_D_olayer_chartrain.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_E_olayer_chartrain.updateAdaGrad(nnRegular, adaAlpha, adaEps);

	}


void writeModel(LStream &outf) {
    _nnlayer_char_hidden.writeModel(outf);
    _nnlayer_char_hidden2.writeModel(outf);
    _A_olayer_chartrain.writeModel(outf);
    _B_olayer_chartrain.writeModel(outf);
    _C_olayer_chartrain.writeModel(outf);
    _D_olayer_chartrain.writeModel(outf);
    _E_olayer_chartrain.writeModel(outf);
    WriteBinary(outf,_A_labelSize);
    WriteBinary(outf,_B_labelSize);
    WriteBinary(outf,_C_labelSize);
    WriteBinary(outf,_D_labelSize);
    WriteBinary(outf,_E_labelSize);
    _chars.writeModel(outf);
    _bichars.writeModel(outf);

    WriteBinary(outf, _charSize);
    WriteBinary(outf, _biCharSize);
    WriteBinary(outf, _charDim);
    WriteBinary(outf, _biCharDim);

    WriteBinary(outf, _charcontext);
    WriteBinary(outf, _charwindow);
    WriteBinary(outf, _charRepresentDim);
    WriteBinary(outf, _charHiddenSize);
    _A_eval.writeModel(outf);
    _B_eval.writeModel(outf);
    _C_eval.writeModel(outf);
    _D_eval.writeModel(outf);
    _E_eval.writeModel(outf);
    WriteBinary(outf, _dropOut);
    WriteBinary(outf, _oovRatio);
    WriteBinary(outf, _buffer);
    WriteBinary(outf, _oovFreq);
    WriteBinary(outf, MAX_SENTENCE_SIZE);
  }

  void loadModel(LStream &inf) {
  	_nnlayer_char_hidden.loadModel(inf);
  	_nnlayer_char_hidden2.loadModel(inf);
    _A_olayer_chartrain.loadModel(inf);
    _B_olayer_chartrain.loadModel(inf);
    _C_olayer_chartrain.loadModel(inf);
    _D_olayer_chartrain.loadModel(inf);
    _E_olayer_chartrain.loadModel(inf);
    ReadBinary(inf,_A_labelSize);
    ReadBinary(inf,_B_labelSize);
    ReadBinary(inf,_C_labelSize);
    ReadBinary(inf,_D_labelSize);
    ReadBinary(inf,_E_labelSize);
    _chars.loadModel(inf);
    _bichars.loadModel(inf);
    ReadBinary(inf, _charSize);
    ReadBinary(inf, _biCharSize);
    ReadBinary(inf, _charDim);
    ReadBinary(inf, _biCharDim);

    ReadBinary(inf, _charcontext);
    ReadBinary(inf, _charwindow);
    ReadBinary(inf, _charRepresentDim);
    ReadBinary(inf, _charHiddenSize);
    _A_eval.loadModel(inf);
    _B_eval.loadModel(inf);
    _C_eval.loadModel(inf);
    _D_eval.loadModel(inf);
    _E_eval.loadModel(inf);
    ReadBinary(inf, _dropOut);
    ReadBinary(inf, _oovRatio);
    ReadBinary(inf, _buffer);
    ReadBinary(inf, _oovFreq);
    ReadBinary(inf, MAX_SENTENCE_SIZE);
  }

  void writeCharEmb(const string& outfile, const Alphabet &charAlpha) {
  	ofstream outstream;
  	outstream.open(outfile.c_str());
  	int charAlphabetSize = charAlpha.size();
  	cout << "Write char Emb to file: " << outfile << ", Size:"<< charAlphabetSize << ", Dim:" << _charDim << endl;
  	Tensor<xpu, 2, dtype> charEmb;
  	charEmb = NewTensor<xpu>(Shape2(1, _charDim), d_zero);
  	for (int idx = 0; idx < charAlphabetSize; ++idx) {
  		_chars.GetEmb(idx, charEmb);
  		outstream << charAlpha.from_id(idx)+" ";
  		for (int idy = 0;idy < _charDim;++idy) {
  			outstream << charEmb[0][idy];
  			if (idy !=_charDim-1) {
  				outstream << " ";
  			} else {
  				outstream << "\n";
  			}
  		}
  	}
  	FreeSpace(&charEmb);
  	outstream.close();
  }

  void writebiCharEmb(const string& outfile, const Alphabet &bicharAlpha) {
  	ofstream outstream;
  	outstream.open(outfile.c_str());
  	int bicharAlphabetSize = bicharAlpha.size();
  	cout << "Write bichar Emb to file: " << outfile << ", Size:"<< bicharAlphabetSize << ", Dim:" << _biCharDim << endl;
  	Tensor<xpu, 2, dtype> bicharEmb;
  	bicharEmb = NewTensor<xpu>(Shape2(1, _biCharDim), d_zero);
  	for (int idx = 0; idx < bicharAlphabetSize; ++idx) {
  		_bichars.GetEmb(idx, bicharEmb);
  		outstream << bicharAlpha.from_id(idx)+" ";
  		for (int idy = 0;idy < _biCharDim;++idy) {
  			outstream << bicharEmb[0][idy];
  			if (idy !=_biCharDim-1) {
  				outstream << " ";
  			} else {
  				outstream << "\n";
  			}
  		}
  	}
  	FreeSpace(&bicharEmb);
  	outstream.close();
  }


  void writeHiddenWeight(const string& outfile) {
  	LStream outf(outfile , "w+");
  	_nnlayer_char_hidden.writeModel(outf);
  	_nnlayer_char_hidden2.writeModel(outf);
  }


void showParameters() {
	cout << "Joint training embedding for two input, hidden layer: 2" << endl;
	cout << "Classifier pamameters: " << endl;
	cout << "MAX_SENTENCE_SIZE = " << MAX_SENTENCE_SIZE << endl;
	cout << "_charSize = " <<  _charSize<< endl;
	cout << "_biCharSize = " <<  _biCharSize << endl;
	cout << "_charDim = " <<  _charDim << endl;
	cout << "_biCharDim = " <<  _biCharDim << endl;
	cout << "_charcontext = " <<  _charcontext << endl;
	cout << "_charwindow = " <<  _charwindow << endl;
	cout << "_charRepresentDim = " <<  _charRepresentDim << endl;
	cout << "_charHiddenSize = " <<  _charHiddenSize << endl;
	cout << "_dropOut = " <<  _dropOut << endl;
	cout << "A_label_size = "<< _A_labelSize << endl;
	cout << "B_label_size = "<< _B_labelSize << endl;
	cout << "C_label_size = "<< _C_labelSize << endl;
	cout << "D_label_size = "<< _D_labelSize << endl;
	cout << "E_label_size = "<< _E_labelSize << endl;
}

public:

	inline void resetEval() {
		_A_eval.reset();
		_B_eval.reset();
		_C_eval.reset();
		_D_eval.reset();
		_E_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

	inline void setOOVRatio(dtype oovRatio) {
		_oovRatio = oovRatio;
	}

	inline void setOOVFreq(dtype oovFreq) {
		_oovFreq = oovFreq;
	}

	inline void setEmbFinetune(bool b_Emb_finetune) {
	   _chars.setEmbFineTune(b_Emb_finetune);
	   _bichars.setEmbFineTune(b_Emb_finetune);
	}

};

#endif /* SRC_TLSTMBeamSearcher_H_ */
