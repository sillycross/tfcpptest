#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

#include "gtest/gtest.h"
#include "common.h"

using namespace tensorflow;

TEST(TFCpp, BenchmarkInvokeOverhead)
{
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        ReleaseAssert(false);
    }

    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), "ReadGraphTest/graph.pb", &graph_def);
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

    const static int numInputs = 10000;

    std::vector<Tensor> arrA, arrB;
    for (int i = 0; i < numInputs; i++)
    {
        arrA.push_back(Tensor(DT_FLOAT, TensorShape()));
        arrA.back().scalar<float>()() = double(rand()) / double(RAND_MAX);
        arrB.push_back(Tensor(DT_FLOAT, TensorShape()));
        arrB.back().scalar<float>()() = double(rand()) / double(RAND_MAX);
    }

    for (int ts = 0; ts < 2; ts++)
    {
        double sum = 0;
        double resultTime;
        {
            AutoTimer timer(&resultTime);
            for (int k = 0; k < numInputs; k++)
            {
                std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
                  { "a", arrA[k] },
                  { "b", arrB[k] },
                };

                std::vector<tensorflow::Tensor> outputs;

                status = session->Run(inputs, {"c"}, {}, &outputs);
                if (!status.ok())
                {
                    std::cout << status.ToString() << "\n";
                    ReleaseAssert(false);
                }

                auto output_c = outputs[0].scalar<float>();
                sum += output_c();
            }
        }

        std::cout << "Accum = " << sum << std::endl;
        std::cout << "Latency per inference is " << resultTime / numInputs * 1000.0 << "ms" << std::endl;
    }

    session->Close();
}
