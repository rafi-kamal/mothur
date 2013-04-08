//
//  rftreenode.cpp
//  Mothur
//
//  Created by Sarah Westcott on 10/2/12.
//  Copyright (c) 2012 Schloss Lab. All rights reserved.
//

#include "rftreenode.hpp"

/***********************************************************************/
RFTreeNode::RFTreeNode(const vector< vector<int> >& baseDataSet,
                       vector<int> bootstrappedTrainingSampleIndices,
                       vector<int> globalDiscardedFeatureIndices,
                       int numFeatures,
                       int numLocalSamples,
                       int numOutputClasses,
                       int generation,
                       int nodeId,
                       float featureStandardDeviationThreshold)

            : baseDataSet(baseDataSet),
            bootstrappedTrainingSampleIndices(bootstrappedTrainingSampleIndices),
            globalDiscardedFeatureIndices(globalDiscardedFeatureIndices),
            numFeatures(numFeatures),
            numLocalSamples(numLocalSamples),
            numOutputClasses(numOutputClasses),
            generation(generation),
            isLeaf(false),
            outputClass(-1),
            nodeId(nodeId),
            testSampleMisclassificationCount(0),
            splitFeatureIndex(-1),
            splitFeatureValue(-1),
            splitFeatureEntropy(-1.0),
            ownEntropy(-1.0),
            featureStandardDeviationThreshold(featureStandardDeviationThreshold),
//            bootstrappedFeatureVectors(numFeatures, vector<int>(numSamples, 0)),
            bootstrappedOutputVector(numLocalSamples, 0),
            leftChildNode(NULL),
            rightChildNode(NULL),
            parentNode(NULL) {
    
    m = MothurOut::getInstance();
    
//    for (int i = 0; i < numSamples; i++) {    // just doing a simple transpose of the matrix
//        if (m->control_pressed) { break; }
//        for (int j = 0; j < numFeatures; j++) { bootstrappedFeatureVectors[j][i] = bootstrappedTrainingSamples[i][j]; }
//    }

    // numSamples is the number of samplas locally in this node, not globally
    for (int i = 0; i < bootstrappedTrainingSampleIndices.size(); i++) { if (m->control_pressed) { break; }
        
        bootstrappedOutputVector[i] = baseDataSet[bootstrappedTrainingSampleIndices[i]][numFeatures];

    }
    
    createLocalDiscardedFeatureList();
    updateNodeEntropy();
}

/***********************************************************************/
void RFTreeNode::getBootstrappedFeatureVector(int i, vector<int>&  bootstrappedFeatureVector){
    for (int j = 0; j < bootstrappedTrainingSampleIndices.size(); j++) {
        bootstrappedFeatureVector[j] = baseDataSet[bootstrappedTrainingSampleIndices[j]][i];
    }
}

/***********************************************************************/
int RFTreeNode::createLocalDiscardedFeatureList(){
    try {
        
        vector<int> bootstrappedFeatureVector(bootstrappedTrainingSampleIndices.size(), 0);
        
        for (int i = 0; i < numFeatures; i++) {
                // TODO: need to check if bootstrappedFeatureVectors.size() == numFeatures, in python code we are using bootstrappedFeatureVectors instead of numFeatures
            if (m->control_pressed) { return 0; }
            
            vector<int>::iterator it = find(globalDiscardedFeatureIndices.begin(), globalDiscardedFeatureIndices.end(), i);
            
            if (it == globalDiscardedFeatureIndices.end()) {                           // NOT FOUND
                getBootstrappedFeatureVector(i, bootstrappedFeatureVector);
                double standardDeviation = m->getStandardDeviation(bootstrappedFeatureVector);  
                if (standardDeviation <= featureStandardDeviationThreshold) { localDiscardedFeatureIndices.push_back(i); }
            }
        }
        
        return 0;
    }
    catch(exception& e) {
        m->errorOut(e, "RFTreeNode", "createLocalDiscardedFeatureList");
        exit(1);
    }  
}
/***********************************************************************/
int RFTreeNode::updateNodeEntropy() {
    try {
        
        vector<int> classCounts(numOutputClasses, 0);
        for (int i = 0; i < bootstrappedOutputVector.size(); i++) {
            classCounts[bootstrappedOutputVector[i]]++;
        }
        int totalClassCounts = accumulate(classCounts.begin(), classCounts.end(), 0);
        double nodeEntropy = 0.0;
        for (int i = 0; i < classCounts.size(); i++) {
            if (m->control_pressed) { return 0; }
            if (classCounts[i] == 0) continue;
            double probability = (double)classCounts[i] / (double)totalClassCounts;
            nodeEntropy += -(probability * log2(probability));
        }
        ownEntropy = nodeEntropy;
        
        return 0;
    }
    catch(exception& e) {
        m->errorOut(e, "RFTreeNode", "updateNodeEntropy");
        exit(1);
    } 
}

/***********************************************************************/
