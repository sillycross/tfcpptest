#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

#include "gtest/gtest.h"
#include "common.h"

using namespace tensorflow;

TEST(TFCpp, BenchmarkMusesRnn)
{
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        ReleaseAssert(false);
    }

    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), "MusesRnn/graph.pb", &graph_def);
    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        ReleaseAssert(false);
    }

    status = session->Create(graph_def);
    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        ReleaseAssert(false);
    }

    const static int inputSize = 11;
    tensorflow::TensorShape inputShape;
    inputShape.AddDim(1);
    inputShape.AddDim(1);
    inputShape.AddDim(inputSize);

    const static int numInputs = 10000;
    std::vector<Tensor> allInputs;
    for (int k = 0; k < numInputs; k++)
    {
        allInputs.push_back(Tensor(DT_FLOAT, inputShape));
        auto a = allInputs.back().flat<float>();
        for (int i = 0; i < inputSize; i++)
        {
            a(i) = double(rand()) / double(RAND_MAX);
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
                std::vector<std::pair<string, tensorflow::Tensor>> input = {
                    { "Placeholder:0", allInputs[k] }
                };

                std::vector<tensorflow::Tensor> outputs;

                status = session->Run(input, {"outputs", "last_states"}, {}, &outputs);
                if (!status.ok())
                {
                    std::cout << status.ToString() << "\n";
                    ReleaseAssert(false);
                }

                sum += outputs[0].flat<float>()(0) + outputs[1].flat<float>()(0);
            }
        }

        std::cout << "Accum = " << sum << std::endl;
        std::cout << "Latency per inference is " << resultTime / numInputs * 1000.0 << "ms" << std::endl;
    }

    session->Close();
}
