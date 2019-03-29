#include "snn.h"
#include <random>
#include <stdexcept>
#include <algorithm>
#include <iostream>

SNN::SNN(unsigned int inputdim, unsigned int layerdim, unsigned int outputdim)
    : inputdim(inputdim),
      layerdim(layerdim),
      outputdim(outputdim),
      labels(layerdim)
{
    for (unsigned int i = 0; i < layerdim; ++i)
        neurons.push_back(STDPNeuron(params, inputdim));
}

unsigned int SNN::operator()(const std::vector<float> &input)
{
    if (input.size() != inputdim)
        throw std::invalid_argument("Input dimension mismatch.");

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    std::vector<std::vector<bool> > input_history(inputdim, std::vector<bool>(tperiod_step));
    std::vector<std::vector<bool> > spike_history(layerdim, std::vector<bool>(tperiod_step));

    for (unsigned int i = 0; i < layerdim; ++i)
        neurons[i].reset();

    for (unsigned int t = 0; t < tperiod_step; ++t) {
        // poisson spike train encoding
        std::vector<bool> spikes(inputdim);
        for (unsigned int i = 0; i < inputdim; ++i) {
            const bool spiked = dist(rng) < input[i] * fr * dt;
            spikes[i] = spiked;
            input_history[i][t] = spiked;
        }

        int max_v_idx = -1;
        float max_v = -1.0;
        for (unsigned int i = 0; i < layerdim; ++i)
            if (neurons[i].run(spikes)) {
                float v = neurons[i].get_v();
                if (v > max_v) {
                    max_v = v;
                    max_v_idx = i;
                }
            }

        if (max_v_idx >= 0) {
            spike_history[max_v_idx][t] = true;

            // inhibition
            for (int i = 0; i < layerdim; ++i)
                if (i != max_v_idx)
                    neurons[i].inhibit();
        }
    }

    std::vector<int> spikecount(layerdim);
    std::transform(spike_history.begin(), spike_history.end(),
                   spikecount.begin(),
                   [](const std::vector<bool> &x) {
                       int sum = 0;
                       for (unsigned int i = 0; i < x.size(); ++i)
                           if (x[i])
                               sum++;
                       return sum;
                   });
    int max_spike_idx = std::distance(spikecount.begin(),
                                      std::max_element(spikecount.begin(),
                                                       spikecount.end()));
    std::cout << "neuron: " << max_spike_idx << " label: " << labels[max_spike_idx] << std::endl;
    return labels[max_spike_idx];
}


void SNN::train(const std::vector<DataItem> &dataset)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (unsigned int idx = 0; idx < dataset.size(); ++idx) {
        if (dataset[idx].in.size() != inputdim)
            throw std::invalid_argument("Input dimension mismatch.");

        std::cout << "train idx " << idx << std::endl;

        std::vector<std::vector<bool> > input_history(inputdim, std::vector<bool>(tperiod_step));
        std::vector<std::vector<bool> > spike_history(layerdim, std::vector<bool>(tperiod_step));

        for (unsigned int i = 0; i < layerdim; ++i)
            neurons[i].reset();

        for (unsigned int t = 0; t < tperiod_step; ++t) {
            // poisson spike train encoding
            std::vector<bool> spikes(inputdim);
            for (unsigned int i = 0; i < inputdim; ++i) {
                const bool spiked = dist(rng) < dataset[idx].in[i] * fr * dt;
                spikes[i] = spiked;
                input_history[i][t] = spiked;
            }

            int max_v_idx = -1;
            float max_v = -1.0;
            for (unsigned int i = 0; i < layerdim; ++i)
                if (neurons[i].run(spikes)) {
                    float v = neurons[i].get_v();
                    if (v > max_v) {
                        max_v = v;
                        max_v_idx = i;
                    }
                }

            if (max_v_idx >= 0) {
                spike_history[max_v_idx][t] = true;

                // inhibition
                for (int i = 0; i < layerdim; ++i)
                    if (i != max_v_idx)
                        neurons[i].inhibit();
            }
        }

        for (unsigned int i = 0; i < layerdim; ++i)
            neurons[i].apply_stdp(input_history, spike_history[i]);
    }
}

void SNN::classify(const std::vector<DataItem> &dataset)
{
    std::vector<std::vector<int> > labelscore(layerdim, std::vector<int>(outputdim));

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (unsigned int idx = 0; idx < dataset.size(); ++idx) {
        if (dataset[idx].in.size() != inputdim)
            throw std::invalid_argument("Input dimension mismatch.");

        std::cout << "classify idx " << idx << std::endl;

        std::vector<std::vector<bool> > input_history(inputdim, std::vector<bool>(tperiod_step));
        std::vector<std::vector<bool> > spike_history(layerdim, std::vector<bool>(tperiod_step));

        for (unsigned int i = 0; i < layerdim; ++i)
            neurons[i].reset();

        for (unsigned int t = 0; t < tperiod_step; ++t) {
            // poisson spike train encoding
            std::vector<bool> spikes(inputdim);
            for (unsigned int i = 0; i < inputdim; ++i) {
                const bool spiked = dist(rng) < dataset[idx].in[i] * fr * dt;
                spikes[i] = spiked;
                input_history[i][t] = spiked;
            }

            int max_v_idx = -1;
            float max_v = -1.0;
            for (unsigned int i = 0; i < layerdim; ++i)
                if (neurons[i].run(spikes)) {
                    float v = neurons[i].get_v();
                    if (v > max_v) {
                        max_v = v;
                        max_v_idx = i;
                    }
                }

            if (max_v_idx >= 0) {
                spike_history[max_v_idx][t] = true;

                // inhibition
                for (int i = 0; i < layerdim; ++i)
                    if (i != max_v_idx)
                        neurons[i].inhibit();
            }
        }

        std::vector<int> spikecount(layerdim);
        std::transform(spike_history.begin(), spike_history.end(),
                       spikecount.begin(),
                       [](const std::vector<bool> &x) {
                           int sum = 0;
                           for (unsigned int i = 0; i < x.size(); ++i)
                               if (x[i])
                                   sum++;
                           return sum;
                       });

        int max_spike_idx = std::distance(spikecount.begin(),
                                          std::max_element(spikecount.begin(),
                                                           spikecount.end()));
        int dataset_label = std::distance(dataset[idx].out.begin(),
                                          std::max_element(dataset[idx].out.begin(),
                                                           dataset[idx].out.end()));
        labelscore[max_spike_idx][dataset_label]++;
    }

    std::transform(labelscore.begin(), labelscore.end(),
                   labels.begin(),
                   [](const std::vector<int> &x) {
                       return std::distance(x.begin(),
                       std::max_element(x.begin(), x.end()));});

    for (int i = 0; i < layerdim; ++i)
        std::cout << labels[i] << ' ';
    std::cout << std::endl;


}
