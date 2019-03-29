#ifndef LIF_H
#define LIF_H

class LIFNeuron
{
public:
    class LIFParams
    {
    public:
        float vinit;
        float tleak;
        float tinhibit;
        float trefrac;
        float tinit;
        float dt;
    };

    LIFNeuron(const LIFParams &params);
    bool stimulate(const float input); // weighted sum
    float get_v() const;
    void inhibit();
    void reset();

private:
    const float vinit;
    const float tleak;
    const float tinhibit;
    const float trefrac;
    const float tinit;
    const float dt;
    float thres;
    float vm;
    float vm_prev;
    unsigned int refractory;
    unsigned int inhibitory;
};


#endif  // LIF_H
