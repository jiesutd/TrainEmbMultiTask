/*
 * CharSegmentor.h
 *
 *  Created on: Dec 24, 2016
 *      Author: Jie
 *      Revised from LSTMClassifier.cpp
 *      Original from meishan zhang
 */

#include "JointTrainemb.h"

#include "Argument_helper.h"

Labeler::Labeler() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  seperateKey = "#";
}

Labeler::~Labeler() {
  // TODO Auto-generated destructor stub
  m_classifier.release();
}

int Labeler::createAlphabet(const vector<Instance>& vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> char_stat;
  hash_map<string, int> bichar_stat;

  m_A_labelAlphabet.clear();

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &labels = pInstance->labels;
    const vector<string>  &chars = pInstance->chars;
    vector<string> features;
    int curInstSize = labels.size();
    int labelId;
    for (int i = 0; i < curInstSize; ++i) {
      labelId = m_A_labelAlphabet.from_string(labels[i]);
      char_stat[chars[i]]++;
      if (i < curInstSize -1) {
        bichar_stat[chars[i]+chars[i+1]]++;
      }
    }
    bichar_stat[nullkey+ chars[0]]++;
    bichar_stat[chars[curInstSize-1]+ nullkey]++;


    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
  cout << "A Label num: " << m_A_labelAlphabet.size() << endl;
  cout << "Total char num: " << char_stat.size() << endl;
  cout << "Total bichar num: " << bichar_stat.size() << endl;

  m_charAlphabet.clear();
  m_charAlphabet.from_string(nullkey);
  m_charAlphabet.from_string(unknownkey);

  m_bicharAlphabet.clear();
  m_bicharAlphabet.from_string(nullkey);
  m_bicharAlphabet.from_string(unknownkey);

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = bichar_stat.begin(); feat_iter != bichar_stat.end(); feat_iter++) {
    if (!m_options.bicharEmbFineTune || feat_iter->second > m_options.bicharCutOff) {
      m_bicharAlphabet.from_string(feat_iter->first);
    }
  }

  cout << "Remain char num: " << m_charAlphabet.size() << endl;
  cout << "Remain bichar num: " << m_bicharAlphabet.size() << endl;
  m_A_labelAlphabet.set_fixed_flag(true);
  m_charAlphabet.set_fixed_flag(true);
  m_bicharAlphabet.set_fixed_flag(true);
  return 0;
}


int Labeler::new_createAlphabet(const vector<Instance>& vecInsts, const string& flag) {
  cout << "Creating Alphabet...," << flag<< endl;

  int numInstance;
  hash_map<string, int> char_stat;
  hash_map<string, int> bichar_stat;

  if (flag == "B") {
    m_B_labelAlphabet.clear();
  } else if (flag == "C") {
    m_C_labelAlphabet.clear();
  } else if (flag == "D") {
    m_D_labelAlphabet.clear();
  } else if (flag == "E") {
    m_E_labelAlphabet.clear();
  } else {
    cout << "new create alphabet error!"<<endl;
  }
  

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &labels = pInstance->labels;
    const vector<string>  &chars = pInstance->chars;
    vector<string> features;
    int curInstSize = labels.size();
    int labelId;
    for (int i = 0; i < curInstSize; ++i) {
      if (flag == "B") {
        labelId = m_B_labelAlphabet.from_string(labels[i]);
      } else if (flag == "C") {
        labelId = m_C_labelAlphabet.from_string(labels[i]);
      } else if (flag == "D") {
        labelId = m_D_labelAlphabet.from_string(labels[i]);
      } else if (flag == "E") {
        labelId = m_E_labelAlphabet.from_string(labels[i]);
      } else {
        cout << "new create alphabet error!"<<endl;
      }
      
      char_stat[chars[i]]++;
      if (i < curInstSize -1) {
        bichar_stat[chars[i]+chars[i+1]]++;
      }
    }
    bichar_stat[nullkey+ chars[0]]++;
    bichar_stat[chars[curInstSize-1]+ nullkey]++;


    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
  if (flag == "B") {
    cout << "B Label num: " << m_B_labelAlphabet.size() << endl;
  } else if (flag == "C") {
    cout << "C Label num: " << m_C_labelAlphabet.size() << endl;
  } else if (flag == "D") {
    cout << "D Label num: " << m_D_labelAlphabet.size() << endl;
  } else if (flag == "E") {
    cout << "E Label num: " << m_E_labelAlphabet.size() << endl;
  } else {
    cout << "new create alphabet error!"<<endl;
  }
  cout << "Total char num: " << char_stat.size() << endl;
  cout << "Total bichar num: " << bichar_stat.size() << endl;

  m_charAlphabet.set_fixed_flag(false);
  m_bicharAlphabet.set_fixed_flag(false);

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = bichar_stat.begin(); feat_iter != bichar_stat.end(); feat_iter++) {
    if (feat_iter->second > m_options.bicharCutOff) {
      m_bicharAlphabet.from_string(feat_iter->first);
    }
  }

  

  if (flag == "B") {
    cout << "Remain char num after B: " << m_charAlphabet.size() << endl;
    cout << "Remain bichar num after B: " << m_bicharAlphabet.size() << endl;
    m_B_labelAlphabet.set_fixed_flag(true);
  } else if (flag == "C") {
    cout << "Remain char num after C: " << m_charAlphabet.size() << endl;
    cout << "Remain bichar num after C: " << m_bicharAlphabet.size() << endl;
    m_C_labelAlphabet.set_fixed_flag(true);
  } else if (flag == "D") {
    cout << "Remain char num after D: " << m_charAlphabet.size() << endl;
    cout << "Remain bichar num after D: " << m_bicharAlphabet.size() << endl;
    m_D_labelAlphabet.set_fixed_flag(true);
  } else if (flag == "E") {
    cout << "Remain char num after E: " << m_charAlphabet.size() << endl;
    cout << "Remain bichar num after E: " << m_bicharAlphabet.size() << endl;
    m_E_labelAlphabet.set_fixed_flag(true);
  } else {
    cout << "new create alphabet error!"<<endl;
  }
  
  m_charAlphabet.set_fixed_flag(true);
  m_bicharAlphabet.set_fixed_flag(true);
  return 0;
}



int Labeler::addTestCharAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding char Alphabet..." << endl;
  int origin_char_size = m_charAlphabet.size();
  int origin_bichar_size = m_bicharAlphabet.size();

  int numInstance;
  hash_map<string, int> char_stat;
  hash_map<string, int> bichar_stat;
  m_charAlphabet.set_fixed_flag(false);
  m_bicharAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &chars = pInstance->chars;

    int curInstSize = chars.size();
    for (int i = 0; i < curInstSize; ++i) {
      char_stat[chars[i]]++;
      if (i < curInstSize -1) {
        bichar_stat[chars[i]+chars[i+1]]++;
      }
    }

    bichar_stat[nullkey+ chars[0]]++;
    bichar_stat[chars[curInstSize-1]+ nullkey]++;

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = bichar_stat.begin(); feat_iter != bichar_stat.end(); feat_iter++) {
    if (!m_options.bicharEmbFineTune || feat_iter->second > m_options.bicharCutOff) {
      m_bicharAlphabet.from_string(feat_iter->first);
    }
  }

  m_charAlphabet.set_fixed_flag(true);
  m_bicharAlphabet.set_fixed_flag(true);

  int char_size = m_charAlphabet.size();
  int bichar_size = m_bicharAlphabet.size();
  cout << endl;
  cout << "New char num: " << char_size << ", " << char_size - origin_char_size << " chars added!" << endl;
  cout << "New bichar num: " << bichar_size << ", " << bichar_size - origin_bichar_size << " bichars added!" << endl;
  return 0;
}


void Labeler::convert2Example(const Instance* pInstance, Example& exam, const Alphabet& m_labelAlphabet) {
  exam.clear();
  const vector<string> &labels = pInstance->labels;
  const vector<string> &chars = pInstance->chars;

  int curInstSize = labels.size();
  assert(curInstSize == chars.size());
  int char_Id;
  int bichar_Id;
  std::string bichar;
  int unknownCharId = m_charAlphabet[unknownkey];
  int unknownbiCharId = m_bicharAlphabet[unknownkey];
  for (int i = 0; i < curInstSize; ++i) {
    string orcale = labels[i];
    int numLabel1s = m_labelAlphabet.size();
    vector<int> curlabels;
    for (int j = 0; j < numLabel1s; ++j) {
      string str = m_labelAlphabet.from_id(j);
      if (str.compare(orcale) == 0)
        curlabels.push_back(1);
      else
        curlabels.push_back(0);
    }
    exam.labels.push_back(curlabels);
  
    char_Id = m_charAlphabet[chars[i]];

    // if (i < curInstSize -1) {
    //   bichar_Id = m_bicharAlphabet[chars[i]+chars[i+1]];
    // } else {
    //   bichar_Id = m_bicharAlphabet[chars[i]+nullkey];
    // }
    bichar = (i == curInstSize-1 ?(chars[i]+nullkey):(chars[i]+chars[i+1]));
    bichar_Id = m_bicharAlphabet[bichar];
    if (char_Id < 0) {
      char_Id = unknownCharId;
    }
    if (bichar_Id < 0) {
      bichar_Id = unknownbiCharId;
    }
    exam.charIds.push_back(char_Id);
    exam.bicharIds.push_back(bichar_Id);
    // cout << "char: " << chars[i] << ", charId: "<< char_Id << ", unkownID:"<< unknownCharId<< endl;
    // cout << "bichar: " << bichar << "  bicharId: "<< bichar_Id<< ", unkownbiID:"<< unknownbiCharId << endl;
  }
  // cout << "origin size: "<< curInstSize << ", new size:" << exam.labels.size() << endl;
  // cout << "exam size: " << exam.charIds.size() << endl;
}


void Labeler::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, const Alphabet& m_labelAlphabet) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];
    Example curExam;
    convert2Example(pInstance, curExam, m_labelAlphabet);
    vecExams.push_back(curExam);

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }
  cout << numInstance << " " << endl;
}


void Labeler::train(const string& atrainFile, const string& adevFile, const string& atestFile,
                    const string& btrainFile, const string& bdevFile, const string& btestFile,
                    const string& ctrainFile, const string& cdevFile, const string& ctestFile,
                    const string& dtrainFile, const string& ddevFile, const string& dtestFile,
                    const string& etrainFile, const string& edevFile, const string& etestFile, 
                    const string& modelFile, const string& partmodelFile, const string& optionFile,
                    const string& charEmbFile, const string& bicharEmbFile) {
  clock_t train_start_time = clock();
  if (optionFile != "")
    m_options.load(optionFile);
  m_options.showOptions();
  vector<Instance> atrainInsts, adevInsts, atestInsts;
  vector<Instance> btrainInsts, bdevInsts, btestInsts;
  vector<Instance> ctrainInsts, cdevInsts, ctestInsts;
  vector<Instance> dtrainInsts, ddevInsts, dtestInsts;
  vector<Instance> etrainInsts, edevInsts, etestInsts;
  static vector<Instance> decodeInstResults;
  // static Instance curDecodeInst;
  bool bCurIterBetter = false;

  m_pipe.readInstances(atrainFile, atrainInsts, m_options.maxInstance);
  if (adevFile != "")
    m_pipe.readInstances(adevFile, adevInsts, m_options.maxInstance);
  if (atestFile != "")
    m_pipe.readInstances(atestFile, atestInsts, m_options.maxInstance);

  m_pipe.readInstances(btrainFile, btrainInsts, m_options.maxInstance);
  if (bdevFile != "")
    m_pipe.readInstances(bdevFile, bdevInsts, m_options.maxInstance);
  if (btestFile != "")
    m_pipe.readInstances(btestFile, btestInsts, m_options.maxInstance);


 m_pipe.readInstances(ctrainFile, ctrainInsts, m_options.maxInstance);
  if (cdevFile != "")
    m_pipe.readInstances(cdevFile, cdevInsts, m_options.maxInstance);
  if (ctestFile != "")
    m_pipe.readInstances(ctestFile, ctestInsts, m_options.maxInstance);


 m_pipe.readInstances(dtrainFile, dtrainInsts, m_options.maxInstance);
  if (ddevFile != "")
    m_pipe.readInstances(ddevFile, ddevInsts, m_options.maxInstance);
  if (dtestFile != "")
    m_pipe.readInstances(dtestFile, dtestInsts, m_options.maxInstance);

 m_pipe.readInstances(etrainFile, etrainInsts, m_options.maxInstance);
  if (edevFile != "")
    m_pipe.readInstances(edevFile, edevInsts, m_options.maxInstance);
  if (etestFile != "")
    m_pipe.readInstances(etestFile, etestInsts, m_options.maxInstance);



  //std::cout << "Training example number: " << trainInsts.size() << std::endl;
  //std::cout << "Dev example number: " << trainInsts.size() << std::endl;
  //std::cout << "Test example number: " << trainInsts.size() << std::endl;

  createAlphabet(atrainInsts);
  new_createAlphabet(btrainInsts,"B");
  new_createAlphabet(ctrainInsts,"C");
  new_createAlphabet(dtrainInsts,"D");
  new_createAlphabet(etrainInsts,"E");

  addTestCharAlpha(adevInsts);
  addTestCharAlpha(atestInsts);

  addTestCharAlpha(bdevInsts);
  addTestCharAlpha(btestInsts);

  addTestCharAlpha(cdevInsts);
  addTestCharAlpha(ctestInsts);

  addTestCharAlpha(ddevInsts);
  addTestCharAlpha(dtestInsts);

  addTestCharAlpha(edevInsts);
  addTestCharAlpha(etestInsts);

  NRMat<dtype> charEmb;
  if (charEmbFile != "") {
    readEmbeddings(m_charAlphabet, charEmbFile, charEmb);
  } else {
    charEmb.resize(m_charAlphabet.size(), m_options.charEmbSize);
    charEmb.randu(1001);
  }

  NRMat<dtype> bicharEmb;
  if (bicharEmbFile != "") {
    readEmbeddings(m_bicharAlphabet, bicharEmbFile, bicharEmb);
  } else {
    bicharEmb.resize(m_bicharAlphabet.size(), m_options.bicharEmbSize);
    bicharEmb.randu(1001);
  }

  m_classifier.setDropValue(m_options.dropProb);
  m_classifier.init(charEmb, bicharEmb, m_options.charcontext, m_options.charHiddenSize, m_A_labelAlphabet.size(), m_B_labelAlphabet.size(), m_C_labelAlphabet.size(), m_D_labelAlphabet.size(), m_E_labelAlphabet.size());
  m_classifier.setEmbFinetune(m_options.charEmbFineTune);

  vector<Example> atrainExamples, adevExamples, atestExamples;
  initialExamples(atrainInsts, atrainExamples, m_A_labelAlphabet);
  initialExamples(adevInsts, adevExamples, m_A_labelAlphabet);
  initialExamples(atestInsts, atestExamples, m_A_labelAlphabet);

  vector<Example> btrainExamples, bdevExamples, btestExamples;
  initialExamples(btrainInsts, btrainExamples, m_B_labelAlphabet);
  initialExamples(bdevInsts, bdevExamples, m_B_labelAlphabet);
  initialExamples(btestInsts, btestExamples, m_B_labelAlphabet);

  vector<Example> ctrainExamples, cdevExamples, ctestExamples;
  initialExamples(ctrainInsts, ctrainExamples, m_C_labelAlphabet);
  initialExamples(cdevInsts, cdevExamples, m_C_labelAlphabet);
  initialExamples(ctestInsts, ctestExamples, m_C_labelAlphabet);


  vector<Example> dtrainExamples, ddevExamples, dtestExamples;
  initialExamples(dtrainInsts, dtrainExamples, m_D_labelAlphabet);
  initialExamples(ddevInsts, ddevExamples, m_D_labelAlphabet);
  initialExamples(dtestInsts, dtestExamples, m_D_labelAlphabet);

  vector<Example> etrainExamples, edevExamples, etestExamples;
  initialExamples(etrainInsts, etrainExamples, m_E_labelAlphabet);
  initialExamples(edevInsts, edevExamples, m_E_labelAlphabet);
  initialExamples(etestInsts, etestExamples, m_E_labelAlphabet);

  vector<Instance>().swap(atrainInsts);
  vector<Instance>().swap(btrainInsts);
  vector<Instance>().swap(ctrainInsts);
  vector<Instance>().swap(dtrainInsts);
  vector<Instance>().swap(etrainInsts);
  


  dtype a_bestDIS = 0;
  dtype b_bestDIS = 0;
  dtype c_bestDIS = 0;
  dtype d_bestDIS = 0;
  dtype e_bestDIS = 0;
  int ainputSize = atrainExamples.size();
  int binputSize = btrainExamples.size();
  int cinputSize = ctrainExamples.size();
  int dinputSize = dtrainExamples.size();
  int einputSize = etrainExamples.size();

  srand(0);
  std::vector<int> aindexes;
  for (int i = 0; i < ainputSize; ++i)
    aindexes.push_back(i);
  std::vector<int> bindexes;
  for (int i = 0; i < binputSize; ++i)
    bindexes.push_back(i);
  std::vector<int> cindexes;
  for (int i = 0; i < cinputSize; ++i)
    cindexes.push_back(i);
  std::vector<int> dindexes;
  for (int i = 0; i < dinputSize; ++i)
    dindexes.push_back(i);
  std::vector<int> eindexes;
  for (int i = 0; i < einputSize; ++i)
    eindexes.push_back(i);


  // int BdivideA = ceil((binputSize+0.0)/ainputSize);
  // int CdivideA = ceil((cinputSize+0.0)/ainputSize);
  // int DdivideA = ceil((dinputSize+0.0)/ainputSize);
  // int EdivideA = ceil((einputSize+0.0)/ainputSize);

  int BdivideA = 10;
  int CdivideA = 3;
  int DdivideA = 1;
  int EdivideA = 0;

  string A_TYPE = "A";
  string B_TYPE = "B";
  string C_TYPE = "C";
  string D_TYPE = "D";
  string E_TYPE = "E";

  static Metric a_eval, a_metric_dev, a_metric_test;
  static Metric b_eval, b_metric_dev, b_metric_test;
  static Metric c_eval, c_metric_dev, c_metric_test;
  static Metric d_eval, d_metric_dev, d_metric_test;
  static Metric e_eval, e_metric_dev, e_metric_test;

  static vector<Example> subExamples;
  static vector<string> subExampleTypes;

  int adevNum = adevExamples.size(), atestNum = atestExamples.size();
  int bdevNum = bdevExamples.size(), btestNum = btestExamples.size();
  int cdevNum = cdevExamples.size(), ctestNum = ctestExamples.size();
  int ddevNum = ddevExamples.size(), dtestNum = dtestExamples.size();
  int edevNum = edevExamples.size(), etestNum = etestExamples.size();

  std::cout << "Train init finished. Total time taken is: " << double(clock() - train_start_time) / CLOCKS_PER_SEC << "s"<< std::endl;
  for (int iter = 0; iter < m_options.maxIter; ++iter) {
    clock_t train_iter_start_time = clock();
    std::cout << "##### Iteration " << iter << std::endl;
    std::cout <<"A total instance: " << ainputSize << std::endl;
    std::cout << "B total instance: " << binputSize << ", B/A = "<< BdivideA<< std::endl;
    std::cout << "C total instance: " << cinputSize << ", C/A = "<< CdivideA<< std::endl;
    std::cout << "D total instance: " << dinputSize << ", D/A = "<< DdivideA<< std::endl;
    std::cout << "E total instance: " << einputSize << ", E/A = "<< EdivideA<< std::endl;

    random_shuffle(aindexes.begin(), aindexes.end());
    random_shuffle(bindexes.begin(), bindexes.end());
    random_shuffle(cindexes.begin(), cindexes.end());
    random_shuffle(dindexes.begin(), dindexes.end());
    random_shuffle(eindexes.begin(), eindexes.end());

    a_eval.reset();
    b_eval.reset();
    c_eval.reset();
    d_eval.reset();
    e_eval.reset();

    clock_t batch_start_time = clock();
    int bupdateIter = 0;
    int cupdateIter = 0;
    int dupdateIter = 0;
    int eupdateIter = 0;
    for (int aupdateIter = 0; aupdateIter < ainputSize; aupdateIter++) {
      subExamples.clear();
      subExampleTypes.clear();

      subExamples.push_back(atrainExamples[aindexes[aupdateIter]]);
      subExampleTypes.push_back(A_TYPE);

      for (int idx = 0; idx < BdivideA; ++idx) {
        if (bupdateIter >= binputSize) {
          continue;
        }
        subExamples.push_back(btrainExamples[bindexes[bupdateIter]]);
        subExampleTypes.push_back(B_TYPE);
        bupdateIter++;
      }

      for (int idx = 0; idx < CdivideA; ++idx) {
        if (cupdateIter >= cinputSize) {
          continue;
        }
        subExamples.push_back(ctrainExamples[cindexes[cupdateIter]]);
        subExampleTypes.push_back(C_TYPE);
        cupdateIter++;
      }

      for (int idx = 0; idx < DdivideA; ++idx) {
        if (dupdateIter >= dinputSize) {
          continue;
        }
        subExamples.push_back(dtrainExamples[dindexes[dupdateIter]]);
        subExampleTypes.push_back(D_TYPE);
        dupdateIter++;
      }

      for (int idx = 0; idx < EdivideA; ++idx) {
        if (eupdateIter >= einputSize) {
          continue;
        }
        subExamples.push_back(etrainExamples[eindexes[eupdateIter]]);
        subExampleTypes.push_back(E_TYPE);
        eupdateIter++;
      }

      dtype cost = m_classifier.train(subExamples, subExampleTypes);

      a_eval.overall_label_count += m_classifier._A_eval.overall_label_count;
      a_eval.correct_label_count += m_classifier._A_eval.correct_label_count;
      b_eval.overall_label_count += m_classifier._B_eval.overall_label_count;
      b_eval.correct_label_count += m_classifier._B_eval.correct_label_count;
      c_eval.overall_label_count += m_classifier._C_eval.overall_label_count;
      c_eval.correct_label_count += m_classifier._C_eval.correct_label_count;
      d_eval.overall_label_count += m_classifier._D_eval.overall_label_count;
      d_eval.correct_label_count += m_classifier._D_eval.correct_label_count;
      e_eval.overall_label_count += m_classifier._E_eval.overall_label_count;
      e_eval.correct_label_count += m_classifier._E_eval.correct_label_count;

      if ((aupdateIter + 1) % m_options.verboseIter == 0) {
        std::cout << "current:" << aupdateIter + 1 << ", Time="<<double(clock() - batch_start_time) / CLOCKS_PER_SEC << ", Cost=" << cost << std::endl;
        std::cout <<"CA=" << a_eval.getAccuracy()<< ", CB=" << b_eval.getAccuracy()<< 
         ", CC=" << c_eval.getAccuracy()<< ", CD=" << d_eval.getAccuracy()<< ", CE=" << e_eval.getAccuracy()
         <<  std::endl;
        batch_start_time = clock();
      }
      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
    }
    clock_t train_iter_end_time = clock();
    std::cout << "Iter "<< iter << " finished. Total time taken is: " << double(train_iter_end_time- train_iter_start_time) / CLOCKS_PER_SEC<< "s" << std::endl;


    if (adevNum > 0) {
      std::cout << "Dev A:" << std::endl;
      DecodeinTraining(adevExamples, adevInsts, decodeInstResults, a_metric_dev, "A", m_options.seg);
      if (!m_options.outBest.empty() && a_metric_dev.getAccuracy() > a_bestDIS) {
        m_pipe.outputAllInstances(adevFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }
      clock_t decode_dev_end_time = clock();
      std::cout << "Decode A dev finished. Total time taken is: " << double(decode_dev_end_time-train_iter_end_time) / CLOCKS_PER_SEC<< "s" << std::endl;
      if (atestNum > 0) {
        std::cout << "Test A:" << std::endl;
        DecodeinTraining(atestExamples, atestInsts,decodeInstResults, a_metric_test, "A", m_options.seg);
        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(atestFile + m_options.outBest, decodeInstResults);
        }
      }
      clock_t decode_test_end_time = clock();
      std::cout << "Decode A test finished. Total time taken is: " << double(decode_test_end_time- decode_dev_end_time) / CLOCKS_PER_SEC<< "s" << std::endl;

      if (m_options.saveIntermediate && a_metric_dev.getAccuracy() > a_bestDIS) {
        std::cout << "Exceeds A best previous performance of " << a_bestDIS << ". Saving model file.." << std::endl;
        a_bestDIS = a_metric_dev.getAccuracy();
        writeModelFile(modelFile);
        writePartModelFile(partmodelFile);
      }
    }

    if (bdevNum > 0) {
      clock_t b_decode_dev_start_time = clock();
      std::cout << "Dev B:" << std::endl;
      DecodeinTraining(bdevExamples, bdevInsts, decodeInstResults, b_metric_dev,  "B", false);
      if (!m_options.outBest.empty() && b_metric_dev.getAccuracy() > b_bestDIS) {
        m_pipe.outputAllInstances(bdevFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }
      clock_t b_decode_dev_end_time = clock();
      std::cout << "Decode B dev finished. Total time taken is: " << double(b_decode_dev_end_time-b_decode_dev_start_time) / CLOCKS_PER_SEC<< "s" << std::endl;
      if (m_options.saveIntermediate && b_metric_dev.getAccuracy() > b_bestDIS) { 
        std::cout << "Exceeds B best previous performance of " << b_bestDIS << ", but save A best model." << endl;
        b_bestDIS = b_metric_dev.getAccuracy();
      }
    }

    if (cdevNum > 0) {
      clock_t c_decode_dev_start_time = clock();
      std::cout << "Dev C:" << std::endl;
      DecodeinTraining(cdevExamples, cdevInsts, decodeInstResults, c_metric_dev,  "C", true);
      if (!m_options.outBest.empty() && c_metric_dev.getAccuracy() > c_bestDIS) {
        m_pipe.outputAllInstances(bdevFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }
      clock_t c_decode_dev_end_time = clock();
      std::cout << "Decode C dev finished. Total time taken is: " << double(c_decode_dev_end_time-c_decode_dev_start_time) / CLOCKS_PER_SEC<< "s" << std::endl;
      if (m_options.saveIntermediate && b_metric_dev.getAccuracy() > b_bestDIS) { 
        std::cout << "Exceeds C best previous performance of " << c_bestDIS << ", but save A best model." << endl;
        c_bestDIS = c_metric_dev.getAccuracy();
      }
    }

    if (ddevNum > 0) {
      clock_t d_decode_dev_start_time = clock();
      std::cout << "Dev D:" << std::endl;
      DecodeinTraining(ddevExamples, ddevInsts, decodeInstResults, d_metric_dev,  "D", false);
      if (!m_options.outBest.empty() && d_metric_dev.getAccuracy() > d_bestDIS) {
        m_pipe.outputAllInstances(ddevFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }
      clock_t d_decode_dev_end_time = clock();
      std::cout << "Decode D dev finished. Total time taken is: " << double(d_decode_dev_end_time-d_decode_dev_start_time) / CLOCKS_PER_SEC<< "s" << std::endl;
      if (m_options.saveIntermediate && b_metric_dev.getAccuracy() > d_bestDIS) { 
        std::cout << "Exceeds D best previous performance of " << d_bestDIS << ", but save A best model." << endl;
        d_bestDIS = d_metric_dev.getAccuracy();
      }
    }

    if (edevNum > 0) {
      clock_t e_decode_dev_start_time = clock();
      std::cout << "Dev E:" << std::endl;
      DecodeinTraining(edevExamples, edevInsts, decodeInstResults, e_metric_dev,  "E", false);
      if (!m_options.outBest.empty() && e_metric_dev.getAccuracy() > e_bestDIS) {
        m_pipe.outputAllInstances(edevFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }
      clock_t e_decode_dev_end_time = clock();
      std::cout << "Decode E dev finished. Total time taken is: " << double(e_decode_dev_end_time-e_decode_dev_start_time) / CLOCKS_PER_SEC<< "s" << std::endl;
      if (m_options.saveIntermediate && e_metric_dev.getAccuracy() > e_bestDIS) { 
        std::cout << "Exceeds E best previous performance of " << e_bestDIS << ", but save A best model." << endl;
        e_bestDIS = e_metric_dev.getAccuracy();
      }
    }
  }
}


void Labeler::DecodeinTraining(const vector<Example>& examples, vector<Instance>& devInsts, vector<Instance>& decodeInstResults, Metric& eval,
                                 const string& exampleType, const bool& seg) {
  Instance curDecodeInst;
  decodeInstResults.clear();
  eval.reset();
  for (int idx = 0; idx < examples.size(); idx++) {
    vector<string> result_labels;
    predict(examples[idx], result_labels, devInsts[idx].chars, exampleType);
    if (seg)
      devInsts[idx].SegEvaluate(result_labels, eval);
    else
      devInsts[idx].Evaluate(result_labels, eval);

    if (!m_options.outBest.empty()) {
      curDecodeInst.copyValuesFrom(devInsts[idx]);
      curDecodeInst.assignLabel(result_labels);
      decodeInstResults.push_back(curDecodeInst);
    }
  }
  eval.print();
}


int Labeler::predict(const Example& example, vector<string>& outputs, const vector<string>& chars, const string& exampleType) {
  assert(example.labels.size() == chars.size());
  vector<int> labelIdx, label2Idx;
  m_classifier.new_decode(example, labelIdx, exampleType);
  outputs.clear();
  string label = "error";
  for (int idx = 0; idx < chars.size(); idx++) {
    if (exampleType == "A") {
       label = m_A_labelAlphabet.from_id(labelIdx[idx]);
    } else if (exampleType == "B") {
       label = m_B_labelAlphabet.from_id(labelIdx[idx]);
    } else if (exampleType == "C") {
       label = m_C_labelAlphabet.from_id(labelIdx[idx]);
    } else if (exampleType == "D") {
       label = m_D_labelAlphabet.from_id(labelIdx[idx]);
    } else if (exampleType == "E") {
       label = m_E_labelAlphabet.from_id(labelIdx[idx]);
    } else {
      cerr << "Predict Error: example type error, " << exampleType << endl;
    }
    outputs.push_back(label);
  }
  return 0;
}


// void Labeler::test(const string& testFile, const string& outputFile, const string& modelFile) {
//   loadModelFile(modelFile);
//   vector<Instance> testInsts;
//   m_pipe.readInstances(testFile, testInsts);

//   vector<Example> testExamples;
//   initialExamples(testInsts, testExamples);

//   int testNum = testExamples.size();
//   vector<Instance> testInstResults;
//   Metric metric_test;
//   metric_test.reset();
//   for (int idx = 0; idx < testExamples.size(); idx++) {
//     vector<string> result_labels;
//     predict(testExamples[idx], result_labels, testInsts[idx].chars);
//     if (m_options.seg) {
//       testInsts[idx].SegEvaluate(result_labels, metric_test);
//     }
//     else {
//       testInsts[idx].Evaluate(result_labels, metric_test);
//     }
//     Instance curResultInst;
//     curResultInst.copyValuesFrom(testInsts[idx]);
//     curResultInst.assignLabel(result_labels);
//     testInstResults.push_back(curResultInst);
//   }
//   std::cout << "test:" << std::endl;
//   metric_test.print();
//   m_pipe.outputAllInstances(outputFile, testInstResults);
// }


void Labeler::readEmbeddings(Alphabet &alpha, const string& inFile, NRMat<dtype>& emb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }
  int unknownId = alpha.from_string(unknownkey);
  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;
  std::cout << "embedding dim is " << wordDim << std::endl;
  emb.resize(alpha.size(), wordDim);
  emb = 0.0;
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = alpha.from_string(curWord);
  hash_set<int> indexers;
  dtype sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      dtype curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      emb[wordId][idx] = curValue;
    }
  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = alpha.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        dtype curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] += curValue;
        emb[wordId][idx] += curValue;
      }
    }
  }
  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      emb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }
  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < alpha.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        emb[id][idx] = emb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << alpha.size() << ", embedding oov ratio is " << oovWords * 1.0 / alpha.size()
      << std::endl;
}


void Labeler::loadModelFile(const string& inputModelFile) {
  std::cout << "Start load model from file: " << inputModelFile << std::endl;
  LStream inf(inputModelFile, "rb");
  m_options.loadModel(inf);
  m_options.showOptions();
  m_charAlphabet.loadModel(inf);
  m_bicharAlphabet.loadModel(inf);
  m_A_labelAlphabet.loadModel(inf);
  m_B_labelAlphabet.loadModel(inf);
  m_classifier.loadModel(inf);
  ReadString(inf, nullkey);
  ReadString(inf, unknownkey);
  ReadString(inf, seperateKey);
  std::cout << "Model has been loaded from file: " << inputModelFile << std::endl;
}


void Labeler::writeModelFile(const string & outputModelFile) {
  std::cout << "Start write model to file: " << outputModelFile << std::endl;
  LStream outf(outputModelFile, "w+");
  m_options.writeModel(outf);
  m_charAlphabet.writeModel(outf);
  m_bicharAlphabet.writeModel(outf);
  m_A_labelAlphabet.writeModel(outf);
  m_B_labelAlphabet.writeModel(outf);
  m_classifier.writeModel(outf);
  WriteString(outf, nullkey);
  WriteString(outf, unknownkey);
  WriteString(outf, seperateKey);
  std::cout << "Model has been written in file: " << outputModelFile << std::endl;
}


void Labeler::writePartModelFile(const string & outputPartModelFile) {
  std::string outlayerFile = outputPartModelFile + ".2h.pmodel";
  std::string outcharEmbFile = outputPartModelFile + ".2h.pchar";
  std::string outbicharEmbFile = outputPartModelFile + ".2h.pbichar";
  std::cout << "Write layer parameter to file: " << outlayerFile<< std::endl;
  m_classifier.writeHiddenWeight(outlayerFile);
  m_classifier.writeCharEmb(outcharEmbFile, m_charAlphabet);
  m_classifier.writebiCharEmb(outbicharEmbFile, m_bicharAlphabet);
}


int main(int argc, char* argv[]) {
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif

  std::string atrainFile = "", adevFile = "", atestFile = "";
  std::string btrainFile = "", bdevFile = "", btestFile = "";
  std::string ctrainFile = "", cdevFile = "", ctestFile = "";
  std::string dtrainFile = "", ddevFile = "", dtestFile = "";
  std::string etrainFile = "", edevFile = "", etestFile = "";
  std::string modelFile = "default.2h.model", partmodelFile = "default";
  std::string charEmbFile = "", bicharEmbFile = "",  optionFile = "";
  std::string outputFile = "";
  bool bTrain = false;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("atrain", "atrainCorpus", "named_string", "a training corpus to train a model, must when training", atrainFile);
  ah.new_named_string("adev", "adevCorpus", "named_string", "a development corpus to train a model, optional when training", adevFile);
  ah.new_named_string("atest", "atestCorpus", "named_string", "a testing corpus to train a model or input file to test a model, optional when training and must when testing", atestFile);
  
  ah.new_named_string("btrain", "btrainCorpus", "named_string", "b training corpus to train a model, must when training", btrainFile);
  ah.new_named_string("bdev", "bdevCorpus", "named_string", "b development corpus to train a model, optional when training", bdevFile);
  ah.new_named_string("btest", "btestCorpus", "named_string", "b testing corpus to train a model or input file to test a model, optional when training and must when testing", btestFile);

  ah.new_named_string("ctrain", "ctrainCorpus", "named_string", "b training corpus to train a model, must when training", ctrainFile);
  ah.new_named_string("cdev", "cdevCorpus", "named_string", "b development corpus to train a model, optional when training", cdevFile);
  ah.new_named_string("ctest", "ctestCorpus", "named_string", "b testing corpus to train a model or input file to test a model, optional when training and must when testing", ctestFile);

  ah.new_named_string("dtrain", "dtrainCorpus", "named_string", "b training corpus to train a model, must when training", dtrainFile);
  ah.new_named_string("ddev", "ddevCorpus", "named_string", "b development corpus to train a model, optional when training", ddevFile);
  ah.new_named_string("dtest", "dtestCorpus", "named_string", "b testing corpus to train a model or input file to test a model, optional when training and must when testing", dtestFile);

  ah.new_named_string("etrain", "etrainCorpus", "named_string", "b training corpus to train a model, must when training", etrainFile);
  ah.new_named_string("edev", "edevCorpus", "named_string", "b development corpus to train a model, optional when training", edevFile);
  ah.new_named_string("etest", "etestCorpus", "named_string", "b testing corpus to train a model or input file to test a model, optional when training and must when testing", etestFile);


  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("pmodel", "partmodelFile", "named_string", "part model file, must when training and testing", partmodelFile);
  ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
  ah.new_named_string("bichar", "bicharEmbFile", "named_string", "pretrained cbihar embedding file to train a model, optional when training", bicharEmbFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Labeler tagger;
  if (bTrain) {
    tagger.train(atrainFile, adevFile, atestFile,
                 btrainFile, bdevFile, btestFile, 
                 ctrainFile, cdevFile, ctestFile, 
                 dtrainFile, ddevFile, dtestFile, 
                 etrainFile, edevFile, etestFile, 
      modelFile, partmodelFile, optionFile, charEmbFile, bicharEmbFile);
  } else {
    // tagger.test(atestFile,btestFile, outputFile, modelFile);
  }

  //test(argv);
  //ah.write_values(std::cout);
#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif
}
