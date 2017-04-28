#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3L.h"

using namespace std;

class Options {
public:

	int wordCutOff;
	int featCutOff;
	int charCutOff;
	int bicharCutOff;
	dtype initRange;
	int maxIter;
	int batchSize;
	dtype adaEps;
	dtype adaAlpha;
	dtype regParameter;
	dtype dropProb;
	dtype delta;
	dtype clip;
	dtype oovRatio;



	int charEmbSize;
	int bicharEmbSize;
	int mapcharEmbSize;
	int charcontext;
	int charHiddenSize;
	bool charEmbFineTune;
	bool bicharEmbFineTune;
	bool mapcharEmbFineTune;


	int verboseIter;
	bool saveIntermediate;
	bool train;
	int maxInstance;
	string outBest;
	bool seg;

	Options() {
		wordCutOff = 4;
		featCutOff = 0;
		charCutOff = 0;
		bicharCutOff = 0;
		initRange = 0.01;
		maxIter = 1000;
		batchSize = 1;
		adaEps = 1e-6;
		adaAlpha = 0.01;
		regParameter = 1e-8;
		dropProb = 0.0;
		delta = 0.2;
		clip = -1.0;
		oovRatio = 0.2;

		charEmbSize = 50;
		bicharEmbSize = 50;
		mapcharEmbSize =256;
		charcontext = 2;
		charHiddenSize = 150;
		charEmbFineTune = false;
		bicharEmbFineTune = false;
		mapcharEmbFineTune = false;

		verboseIter = 100;
		saveIntermediate = true;
		train = false;
		maxInstance = -1;
		outBest = "";
		seg = true;

	}

	virtual ~Options() {

	}

	void setOptions(const vector<string> &vecOption) {
		int i = 0;
		for (; i < vecOption.size(); ++i) {
			pair<string, string> pr;
			string2pair(vecOption[i], pr, '=');
			if (pr.first == "wordCutOff")
				wordCutOff = atoi(pr.second.c_str());
			if (pr.first == "featCutOff")
				featCutOff = atoi(pr.second.c_str());
			if (pr.first == "charCutOff")
				charCutOff = atoi(pr.second.c_str());
			if (pr.first == "bicharCutOff")
				bicharCutOff = atoi(pr.second.c_str());
			if (pr.first == "initRange")
				initRange = atof(pr.second.c_str());
			if (pr.first == "maxIter")
				maxIter = atoi(pr.second.c_str());
			if (pr.first == "batchSize")
				batchSize = atoi(pr.second.c_str());
			if (pr.first == "adaEps")
				adaEps = atof(pr.second.c_str());
			if (pr.first == "adaAlpha")
				adaAlpha = atof(pr.second.c_str());
			if (pr.first == "regParameter")
				regParameter = atof(pr.second.c_str());
			if (pr.first == "dropProb")
				dropProb = atof(pr.second.c_str());
			if (pr.first == "delta")
				delta = atof(pr.second.c_str());
			if (pr.first == "clip")
				clip = atof(pr.second.c_str());
			if (pr.first == "oovRatio")
				oovRatio = atof(pr.second.c_str());

			if (pr.first == "charEmbSize")
				charEmbSize = atoi(pr.second.c_str());
			if (pr.first == "bicharEmbSize")
				bicharEmbSize = atoi(pr.second.c_str());
			if (pr.first == "mapcharEmbSize")
				mapcharEmbSize = atoi(pr.second.c_str());
			if (pr.first == "charcontext")
				charcontext = atoi(pr.second.c_str());
			if (pr.first == "charHiddenSize")
				charHiddenSize = atoi(pr.second.c_str());
			if (pr.first == "charEmbFineTune")
				charEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "bicharEmbFineTune")
				bicharEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "mapcharEmbFineTune")
				mapcharEmbFineTune = (pr.second == "true") ? true : false;

			if (pr.first == "verboseIter")
				verboseIter = atoi(pr.second.c_str());
			if (pr.first == "train")
				train = (pr.second == "true") ? true : false;
			if (pr.first == "saveIntermediate")
				saveIntermediate = (pr.second == "true") ? true : false;
			if (pr.first == "maxInstance")
				maxInstance = atoi(pr.second.c_str());
			if (pr.first == "outBest")
				outBest = pr.second;
			if (pr.first == "seg")
				seg = (pr.second == "true") ? true : false;
		}
	}

	void showOptions() {
		std::cout << "wordCutOff = " << wordCutOff << std::endl;
		std::cout << "featCutOff = " << featCutOff << std::endl;
		std::cout << "charCutOff = " << charCutOff << std::endl;
		std::cout << "bicharCutOff = " << bicharCutOff << std::endl;
		std::cout << "initRange = " << initRange << std::endl;
		std::cout << "maxIter = " << maxIter << std::endl;
		std::cout << "batchSize = " << batchSize << std::endl;
		std::cout << "adaEps = " << adaEps << std::endl;
		std::cout << "adaAlpha = " << adaAlpha << std::endl;
		std::cout << "regParameter = " << regParameter << std::endl;
		std::cout << "dropProb = " << dropProb << std::endl;
		std::cout << "delta = " << delta << std::endl;
		std::cout << "clip = " << clip << std::endl;
		std::cout << "oovRatio = " << oovRatio << std::endl;

		std::cout << "charEmbSize = " << charEmbSize << std::endl;
		std::cout << "bicharEmbSize = " << bicharEmbSize << std::endl;
		std::cout << "mapcharEmbSize = " << mapcharEmbSize << std::endl;
		std::cout << "charcontext = " << charcontext << std::endl;
		std::cout << "charHiddenSize = " << charHiddenSize << std::endl;
	
		std::cout << "charEmbFineTune = " << charEmbFineTune << std::endl;
		std::cout << "bicharEmbFineTune = " << bicharEmbFineTune << std::endl;
		std::cout << "mapcharEmbFineTune = " << mapcharEmbFineTune << std::endl;

		std::cout << "verboseIter = " << verboseIter << std::endl;
		std::cout << "saveItermediate = " << saveIntermediate << std::endl;
		std::cout << "train = " << train << std::endl;
		std::cout << "maxInstance = " << maxInstance << std::endl;
		std::cout << "outBest = " << outBest << std::endl;
		std::cout << "seg = " << seg << std::endl;
	}

	void load(const std::string& infile) {
		ifstream inf;
		inf.open(infile.c_str());
		vector<string> vecLine;
		while (1) {
			string strLine;
			if (!my_getline(inf, strLine)) {
				break;
			}
			if (strLine.empty())
				continue;
			vecLine.push_back(strLine);
		}
		inf.close();
		setOptions(vecLine);
	}


void writeModel(LStream &outf) {
	WriteBinary(outf, wordCutOff);
    WriteBinary(outf, featCutOff);
    WriteBinary(outf, charCutOff);
    WriteBinary(outf, bicharCutOff);
    WriteBinary(outf, initRange);
    WriteBinary(outf, maxIter);
    WriteBinary(outf, batchSize);
    WriteBinary(outf, adaEps);
    WriteBinary(outf, adaAlpha);
    WriteBinary(outf, regParameter);
    WriteBinary(outf, dropProb);
    WriteBinary(outf, delta);
    WriteBinary(outf, clip);
    WriteBinary(outf, oovRatio);

    WriteBinary(outf, charEmbSize);
    WriteBinary(outf, bicharEmbSize);
    WriteBinary(outf, mapcharEmbSize);
    WriteBinary(outf, charcontext);

    WriteBinary(outf, charHiddenSize);
    WriteBinary(outf, charEmbFineTune);
    WriteBinary(outf, bicharEmbFineTune);
    WriteBinary(outf, mapcharEmbFineTune);
    WriteBinary(outf, verboseIter);
    WriteBinary(outf, saveIntermediate);
    WriteBinary(outf, train);
    WriteBinary(outf, maxInstance);
    WriteString(outf, outBest);
    WriteBinary(outf, seg);
  }

  void loadModel(LStream &inf) {
  	ReadBinary(inf, wordCutOff);
    ReadBinary(inf, featCutOff);
    ReadBinary(inf, charCutOff);
    ReadBinary(inf, bicharCutOff);
    ReadBinary(inf, initRange);
    ReadBinary(inf, maxIter);
    ReadBinary(inf, batchSize);
    ReadBinary(inf, adaEps);
    ReadBinary(inf, adaAlpha);
    ReadBinary(inf, regParameter);
    ReadBinary(inf, dropProb);
    ReadBinary(inf, delta);
    ReadBinary(inf, clip);
    ReadBinary(inf, oovRatio);

    ReadBinary(inf, charEmbSize);
    ReadBinary(inf, bicharEmbSize);
    ReadBinary(inf, mapcharEmbSize);
    ReadBinary(inf, charcontext);

    ReadBinary(inf, charHiddenSize);
    ReadBinary(inf, charEmbFineTune);
    ReadBinary(inf, bicharEmbFineTune);
    ReadBinary(inf, mapcharEmbFineTune);
    ReadBinary(inf, verboseIter);
    ReadBinary(inf, saveIntermediate);
    ReadBinary(inf, train);
    ReadBinary(inf, maxInstance);
    ReadString(inf, outBest);
    ReadBinary(inf, seg);
  }
};

#endif

