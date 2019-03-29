#include "stdp.h"
#include <random>
#include <algorithm>

STDPNeuron::STDPNeuron(const LIFNeuron::LIFParams &params, unsigned int dim)
    : LIFNeuron(params),
      dim(dim),
      weights(dim)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    std::generate_n(weights.begin(), dim,
                    [&dist, &rng](){return dist(rng);});
}

bool STDPNeuron::run(const std::vector<bool> &input)
{
    float input_current = 0.0f;
    for (unsigned int i = 0; i < input.size(); ++i)
        if (input[i])
            input_current += weights[i];

    return LIFNeuron::stimulate(input_current);
}


void STDPNeuron::apply_stdp(const std::vector<std::vector<bool> > &input_history,
                            const std::vector<bool> &spike_history)
{
    int tperiod_step = input_history[0].size();

    for (int t = 0; t < tperiod_step; ++t)
        if (spike_history[t])
            for (unsigned int i = 0; i < input_history.size(); ++i) {
                for (int j = std::max(0, t - 10); j < t; ++j)
                    if (input_history[i][j])
                        weights[i] += 0.1 * std::exp((j - t));
                for (int j = t + 1; j < std::min(t + 10, tperiod_step); ++j)
                    if (input_history[i][j])
                        weights[i] -= 0.03 * std::exp((t - j));
                if (weights[i] < 0)
                    weights[i] = 0;
            }
}

