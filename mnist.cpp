#include "mnist.h"
#include <fstream>
#include <stdexcept>
#include <memory>

MNIST::MNIST(const char *imagefile, const char *labelfile)
{
    char p[4] = {0};

    // read image
    std::ifstream imgstream(imagefile, std::ios::in | std::ios::binary);

    // magic number
    imgstream.read(p, 4);
    if (p2num(p) != 0x803)
        throw std::invalid_argument("Magic number mismatch (image)");

    // size
    imgstream.read(p, 4);
    size = p2num(p);

    // rows
    imgstream.read(p, 4);
    rows = p2num(p);

    // cols
    imgstream.read(p, 4);
    cols = p2num(p);

    // images
    std::unique_ptr<char[]> buffer(new char[rows * cols]);
    for (unsigned int i = 0; i < size; ++i) {
        imgstream.read(buffer.get(), rows * cols);
        std::vector<float> img(rows * cols);
        for (unsigned int j = 0; j < rows * cols; ++j)
            img[j] = static_cast<unsigned char>(buffer[j]) / 255.0f;
        images.push_back(img);
    }

    // read labels
    std::ifstream lblstream(labelfile, std::ios::in | std::ios::binary);

    // magic number
    lblstream.read(p, 4);
    if (p2num(p) != 0x801)
        throw std::invalid_argument("Magic number mismatch (label)");

    // size
    lblstream.read(p, 4);
    if (p2num(p) != size)
        throw std::invalid_argument("File size mismatch");

    // labels
    for (unsigned int i = 0; i < size; ++i) {
        lblstream.read(p, 1);
        labels.push_back(static_cast<unsigned int>(p[0]));
    }
}

uint32_t MNIST::p2num(char p[4])
{
    uint32_t ret = 0;
    ret |= (p[0] & 0xff) << 24;
    ret |= (p[1] & 0xff) << 16;
    ret |= (p[2] & 0xff) << 8;
    ret |= p[3] & 0xff;
    return ret;
}

std::vector<DataItem> MNIST::getdataset() const
{
    std::vector<DataItem> ret(size);
    for (unsigned int i = 0; i < size; ++i) {
        ret[i].in = images[i];
        ret[i].out = std::vector<float>(LABEL_MAX);
        ret[i].out[labels[i]] = 1.0f;
    }
    return ret;
}

unsigned int MNIST::getsize() const
{
    return size;
}

unsigned int MNIST::getrows() const
{
    return rows;
}

unsigned int MNIST::getcols() const
{
    return cols;
}

std::vector<float> MNIST::getimage(int index) const
{
    return images[index];
}

unsigned int MNIST::getlabel(int index) const
{
    return labels[index];
}
