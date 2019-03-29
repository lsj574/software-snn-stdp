#ifndef MNIST_H
#define MNIST_H

#include "dataitem.h"
#include <vector>
#include <cstdint>

class MNIST
{
public:
    MNIST(const char *imagefile, const char *labelfile);
    std::vector<DataItem> getdataset() const;
    unsigned int getsize() const;
    unsigned int getrows() const;
    unsigned int getcols() const;
    std::vector<float> getimage(int index) const;
    unsigned int getlabel(int index) const;
    static const unsigned int LABEL_MAX = 10;

private:
    static uint32_t p2num(char p[4]);
    unsigned int size;
    unsigned int rows;
    unsigned int cols;
    std::vector<std::vector<float> > images;
    std::vector<unsigned int> labels;
};

#endif  // MNIST_H
