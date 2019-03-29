#include "mnist.h"
#include "snn.h"
#include <iostream>

int main(int argc, char *argv[])
{
    MNIST trainset("../MNIST/train-images.idx3-ubyte", "../MNIST/train-labels.idx1-ubyte");
    MNIST testset("../MNIST/t10k-images.idx3-ubyte", "../MNIST/t10k-labels.idx1-ubyte");

    const unsigned int inputdim = trainset.getrows() * trainset.getcols();
    const unsigned int layerdim = 300;
    const unsigned int outputdim = MNIST::LABEL_MAX;

    // FIXME: short version
    std::vector<DataItem> dataset2(trainset.getdataset());
    std::vector<DataItem> dataset(dataset2.begin(), dataset2.begin()+1000);

    SNN snn(inputdim, layerdim, outputdim);
    std::cout << "training..." << std::endl;
    snn.train(dataset);
    std::cout << "classifying..." << std::endl;
    snn.classify(dataset);

    int count = 0;
    for (unsigned int i = 0; i < testset.getsize(); ++i)
        if (snn(testset.getimage(i)) == testset.getlabel(i))
            count++;

    std::cout << count << " per " << testset.getsize() << std::endl;

}
