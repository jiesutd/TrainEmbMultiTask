/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_TRAINEMB_H_
#define SRC_TRAINEMB_H_


#include "N3L.h"
#include "model/JointCharTrain.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Labeler {

public:
  std::string nullkey;
  std::string unknownkey;
  std::string seperateKey;

public:

  Alphabet m_A_labelAlphabet;
  Alphabet m_B_labelAlphabet;
  Alphabet m_C_labelAlphabet;
  Alphabet m_D_labelAlphabet;
  Alphabet m_E_labelAlphabet;

  Alphabet m_charAlphabet;
  Alphabet m_bicharAlphabet;


public:
  Options m_options;

  Pipe m_pipe;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
  Classifier<cpu> m_classifier;
#endif

public:
  void readEmbeddings(Alphabet &alpha, const string& inFile, NRMat<dtype>& wordEmb);

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  int new_createAlphabet(const vector<Instance>& vecInsts, const string& flag);
  // int addTestWordAlpha(const vector<Instance>& vecInsts);
  int addTestCharAlpha(const vector<Instance>& vecInsts);
  // int addTestTagAlpha(const vector<Instance>& vecInsts);

  // void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
  // void extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void convert2Example(const Instance* pInstance, Example& exam, const Alphabet& m_labelAlphabet);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, const Alphabet& m_labelAlphabet);

public:
  void train(const string& atrainFile, const string& adevFile, const string& atestFile,
              const string& btrainFile, const string& bdevFile, const string& btestFile,
              const string& ctrainFile, const string& cdevFile, const string& ctestFile,
              const string& dtrainFile, const string& ddevFile, const string& dtestFile,
              const string& etrainFile, const string& edevFile, const string& etestFile,
             const string& modelFile,const string& partmodelFile, const string& optionFile, const string& charEmbFile, const string& bicharEmbFile);
  int predict(const Example& example, vector<string>& outputs, const vector<string>& words, const string& exampleType);
  // void test(const string& testFile, const string& outputFile, const string& modelFile);
  void DecodeinTraining(const vector<Example>& examples, vector<Instance>& devInsts,vector<Instance>& decodeInstResults, Metric& eval,
                          const string& exampleType, const bool& seg);
  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);
  
  void writePartModelFile(const string& outputModelFile);
};

#endif /* SRC_NNCRF_H_ */
