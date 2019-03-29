#ifndef SNN_H
#define SNN_H

#include "dataitem.h"
#include "stdp.h"

class SNN
{
public:
    SNN(unsigned int inputdim, unsigned int layerdim, unsigned int outputdim);
    unsigned int operator()(const std::vector<float> &input);
    void train(const std::vector<DataItem> &dataset);
    void classify(const std::vector<DataItem> &dataset);

private:
    const unsigned int inputdim;
    const unsigned int layerdim;
    const unsigned int outputdim;
    std::vector<STDPNeuron> neurons;
    std::vector<unsigned int> labels;
    const float fr = 20.0f;
    const float dt = 0.005f;
    const float tperiod = 0.5f;
    const unsigned int tperiod_step = 500;
    const LIFNeuron::LIFParams params =
        { 0.0f, 0.5f, 0.000f, 0.02f, 70.0f * dt, dt};
    // FIXME: tinhibit, tinit
};



#endif  // SNN_H
