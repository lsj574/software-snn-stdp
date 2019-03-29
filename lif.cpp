#include "lif.h"

LIFNeuron::LIFNeuron(const LIFParams &params)
    : vinit(params.vinit),
      tleak(params.tleak),
      tinhibit(params.tinhibit),
      trefrac(params.trefrac),
      tinit(params.tinit),
      dt(params.dt),
      thres(tinit),
      vm(0),
      vm_prev(0),
      refractory(0),
      inhibitory(0) {}

bool LIFNeuron::stimulate(const float input)
{
    bool isspike = false;
    vm = vinit;

    if (inhibitory == 0 && refractory == 0) {
        vm = vm_prev + (-vm_prev / tleak + input) * dt;
        if (vm > thres) {
            isspike = true;
            refractory = static_cast<unsigned int>(trefrac / dt);
        }
    }
    else {
        if (refractory > 0)
            --refractory;
        if (inhibitory > 0)
            --inhibitory;

        // input = 0
        // vm = vm_prev + (-vm_prev / tleak) * dt;
    }

    vm_prev = vm;

    return isspike;
}

float LIFNeuron::get_v() const
{
    return vm;
}

void LIFNeuron::inhibit()
{
    inhibitory = static_cast<unsigned int>(tinhibit / dt);
}

void LIFNeuron::reset()
{
    vm = 0;
}
