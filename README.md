# Investigating performance of real-time Tensorflow inference

The benchmarked RNN network (2-layer, 128 cells each) takes about 547K 32-bit FMA operations to do one inference.
We investigated the latency of doing one single inference (no batching) on CPU.
We investigated the following possibilities:
* Using Tensorflow Python API.
* Using Tensorflow C++ API.
* Using Tensorflow ahead-of-time compiled binary. 

# Result numbers

The result is as follows.
* It takes about 500us to invoke Tensorflow from Python to do one inference of the above-mentioned RNN network.
* It takes about 160us to invoke Tensorflow from C++ to do do one inference of the above-mentioned RNN network.
* It takes about 85us to do one inference using the Tensorflow AOT Compiled Binary of the RNN network from C++.

Additionally:
* It takes about 130us to invoke Tensorflow from Python to do a trivial integer multiplication.
* It takes about 8us to invoke Tensorflow from C++ to do a trivial integer multiplication.
* It takes about 100us to do 547K 32-bit FMA ops using AVX2 on my CPU, but my CPU is actually performing slightly worse with FMA than without, so this number is not a quantatively precise lower-bounding of the cost of one inference, though it should be close. 

# Analysis and Lessions learned:

* Invoking Tensorflow functions from Python incurs a significant overhead. There is 130us overhead even for a trivial Tensorflow call. Doing NN inference incurs an additional overhead of 500us - 160us (C++ perf) - 130us = 210us in Python, which probably because more Python codepaths are executed.
* Invoking Tensorflow from C++ has significantly lower overhead than from Python (we saved 340us by doing so). However, there are still non-negligible overheads even in C++ API; indeed, switching to Tensorflow AOT Compiled Binary saves another 75us. The source of the overhead in C++ API is not fully clear to me, but perf report suggests that at least 35% of the time is wasted in thread pool. 
* Tensorflow AOT compilation is actually doing a great job. It is utilizing the hardware resource close to perfect, if not perfect: doing the same amount of FLOPS using a trivial vectorized loop takes about the same time. 

