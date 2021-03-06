#ifndef READPHYLIP_H
#define READPHYLIP_H
/*
 *  readphylip.h
 *  Mothur
 *
 *  Created by Sarah Westcott on 4/21/09.
 *  Copyright 2009 Schloss Lab UMASS Amherst. All rights reserved.
 *
 */

#include "readmatrix.hpp"

/******************************************************/

class ReadPhylipMatrix : public ReadMatrix {
	
public:
	ReadPhylipMatrix(string);
	ReadPhylipMatrix(string, bool);
	~ReadPhylipMatrix();
	int read(NameAssignment*);
    int read(CountTable*);
private:
	ifstream fileHandle;
	string distFile;
};

/******************************************************/

#endif
