#include "gtest/gtest.h"

#include "MusesRnnTfCompiled/graph.h"
#include "common.h"

TEST(TFCpp, BenchmarkAotCompiled)
{
    Graph g(Graph::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);

    const static int inputSize = 11;
    const static int outputSize = 256;
    const static int numInputs = 10000;

    std::vector<std::vector<float>> allInputs, allOutputs;
    allInputs.resize(numInputs);
    allOutputs.resize(numInputs);
    for (int k = 0; k < numInputs; k++)
    {
        allOutputs[k].resize(outputSize);
        allInputs[k].resize(inputSize);
        for (int i = 0; i < inputSize; i++)
        {
            allInputs[k][i] = double(rand()) / double(RAND_MAX);
        }
    }

    for (int ts = 0; ts < 2; ts++)
    {
        double sum = 0;
        double resultTime;
        {
            AutoTimer timer(&resultTime);
            for (int k = 0; k < numInputs; k++)
            {
                g.set_arg0_data(allInputs[k].data());
                ReleaseAssert(g.Run());
                float* res = g.result0_data();
                // memcpy(allOutputs[k].data(), res, sizeof(float) * outputSize);
                sum += res[0] + res[1];
            }
        }

        std::cout << "Accum = " << sum << std::endl;
        std::cout << "Latency per inference is " << resultTime / numInputs * 1000.0 << "ms" << std::endl;
    }

}
