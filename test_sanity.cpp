#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include <iostream>

#include "gtest/gtest.h"

#include "common.h"

using namespace std;
using namespace tensorflow;

TEST(TFCppSanity, CreateSession)
{
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        cout << status.ToString() << "\n";
        ReleaseAssert(false);
    }
    else
    {
    	cout << "Session successfully created.\n";
    }
} 
