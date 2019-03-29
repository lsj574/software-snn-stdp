#ifndef STDP_H
#define STDP_H

#include "lif.h"
#include <vector>

class STDPNeuron : public LIFNeuron
{
public:
    STDPNeuron(const LIFNeuron::LIFParams &params, unsigned int dim);
    bool run(const std::vector<bool> &input);
    void apply_stdp(const std::vector<std::vector<bool> > &input_history,
                    const std::vector<bool> &spike_history);
    // TODO
    void apply_homeo();

private:
    const unsigned int dim;
    std::vector<float> weights; // pre-post
};


#endif  // STDP_H
