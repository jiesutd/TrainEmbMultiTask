/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "Feature.h"

using namespace std;

class Example {

public:
  vector<vector<int> > labels;
  vector<int> charIds;
  vector<int> bicharIds;

public:
  Example()
  {

  }
  virtual ~Example()
  {

  }

  void clear()
  {
    labels.clear();
    charIds.clear();
    bicharIds.clear();
  }


};

#endif /* SRC_EXAMPLE_H_ */
