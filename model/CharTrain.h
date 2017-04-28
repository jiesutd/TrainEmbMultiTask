/*
 * CharSegmentor.h
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
	UniLayer<xpu> _olayer_chartrain;
	int _labelSize = 0;
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

	Metric _eval;

	dtype _dropOut;

	dtype _oovRatio;
	int _buffer;

	int _oovFreq = 0;

	int MAX_SENTENCE_SIZE = 512;

public:

	inline void init(const NRMat<dtype>& charEmb, const NRMat<dtype>& bicharEmb, int charcontext, int charHiddenSize, int labelSize) {

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
		
		_labelSize = labelSize;
		_olayer_chartrain.initial(_labelSize, _charHiddenSize, true, 60, 2);

		showParameters();
	}

	inline void release() {
		//_splayer_output.release();
		_chars.release();
		_bichars.release();
		_nnlayer_char_hidden.release();
		_nnlayer_char_hidden2.release();
		_olayer_chartrain.release();

	}

	dtype train(const vector<Example>& examples) {
		_eval.reset();
		dtype cost = 0.0;
		int batch_size = examples.size();
		for (int idx = 0; idx < examples.size(); idx++) {
			cost += trainOneExample(examples[idx],batch_size );
		}
		return cost;
	}


	dtype trainOneExample(const Example& example, int batch_size) {
		srand(batch_size);
		int length = example.charIds.size();
		if (length >= MAX_SENTENCE_SIZE) {
			cout << "sentence size is "<<length<< " larger than " << MAX_SENTENCE_SIZE <<endl;
			return 0.0;
		}
		dtype cost = 0.0;
		static DenseFeatureCharWithMap<xpu> charFeat;

		charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_labelSize, _buffer, true);
		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = example.charIds[idx];
			charFeat._bicharIds[idx] = example.bicharIds[idx];
		}
		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);
			dropoutcol(charFeat._charpreMask[idx], _dropOut);
			charFeat._charpre[idx] = charFeat._charpre[idx] * charFeat._charpreMask[idx];
		}
		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);
		_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
		_nnlayer_char_hidden2.ComputeForwardScore(charFeat._charHidden, charFeat._charHidden2);
		_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);


		cost += newsoftmax_loss(charFeat._charOut, example.labels, charFeat._charOut_Loss, _eval, batch_size);

		_olayer_chartrain.ComputeBackwardLoss(charFeat._charHidden2, charFeat._charOut, charFeat._charOut_Loss, charFeat._charHidden2_Loss);
		_nnlayer_char_hidden2.ComputeBackwardLoss(charFeat._charHidden, charFeat._charHidden2, charFeat._charHidden2_Loss, charFeat._charHidden_Loss);
		_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput, charFeat._charHidden, charFeat._charHidden_Loss, charFeat._charInput_Loss);
		windowlized_backward(charFeat._charpre_Loss, charFeat._charInput_Loss, _charcontext);
		charFeat._charpre_Loss = charFeat._charpre_Loss * charFeat._charpreMask;
		for(int idx = 0; idx < length; idx++){
			unconcat(charFeat._charprime_Loss[idx], charFeat._bicharprime_Loss[idx], charFeat._charpre_Loss[idx]);
			_chars.EmbLoss(charFeat._charIds[idx], charFeat._charprime_Loss[idx]);
			_bichars.EmbLoss(charFeat._bicharIds[idx], charFeat._bicharprime_Loss[idx]);
		}

		charFeat.clear();
		if (_eval.getAccuracy() < 0) {
	      std::cout << "strange" << std::endl;
	    }
		return cost;
	}

	
	bool decode(const Example& example, vector<int>& results) {
		_eval.reset();
		dtype cost = 0.0;
		int length = example.charIds.size();
		if (length >= MAX_SENTENCE_SIZE) {
			cout << "sentence size is "<<length<< " larger than " << MAX_SENTENCE_SIZE <<endl;
			return 0.0;
		}
		static DenseFeatureCharWithMap<xpu> charFeat;

		charFeat.init(length, _charDim, _biCharDim, 0, 0, _charcontext, _charHiddenSize,_labelSize, _buffer, true);
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
		_olayer_chartrain.ComputeForwardScore(charFeat._charHidden2, charFeat._charOut);

		newsoftmax_predict(charFeat._charOut, results);

		charFeat.clear();
		return true;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps, dtype clip = -1.0) {
		if(clip > 0.0) {
			dtype norm = 0.0;
			norm += _chars.squarenormAll();
			norm += _bichars.squarenormAll();

			norm += _nnlayer_char_hidden.squarenormAll();
			norm += _nnlayer_char_hidden2.squarenormAll();

			norm += _olayer_chartrain.squarenormAll();
			
			if(norm > clip * clip){
				dtype scale = clip/sqrt(norm);
				_chars.scaleGrad(scale);
				_bichars.scaleGrad(scale);
				_nnlayer_char_hidden.scaleGrad(scale);
				_nnlayer_char_hidden2.scaleGrad(scale);
				_olayer_chartrain.scaleGrad(scale);
			}
		}
		_chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_bichars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_olayer_chartrain.updateAdaGrad(nnRegular, adaAlpha, adaEps);

	}


void writeModel(LStream &outf) {
    _nnlayer_char_hidden.writeModel(outf);
    _nnlayer_char_hidden2.writeModel(outf);
    _olayer_chartrain.writeModel(outf);
    WriteBinary(outf,_labelSize);
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
    _eval.writeModel(outf);
    WriteBinary(outf, _dropOut);
    WriteBinary(outf, _oovRatio);
    WriteBinary(outf, _buffer);
    WriteBinary(outf, _oovFreq);
    WriteBinary(outf, MAX_SENTENCE_SIZE);
  }

  void loadModel(LStream &inf) {
  	_nnlayer_char_hidden.loadModel(inf);
  	_nnlayer_char_hidden2.loadModel(inf);
    _olayer_chartrain.loadModel(inf);
    ReadBinary(inf,_labelSize);
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
    _eval.loadModel(inf);
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


void showParameters() {
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
	cout << "chartrain_label_size = "<< _labelSize << endl;
}

public:

	inline void resetEval() {
		_eval.reset();
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
