nvcc -O3 -arch=sm_90 -o test_correctness_1 test_correctness_1.cu
./test_correctness_1 >> ./logs/test_correctness_1.log 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_2 test_correctness_2.cu
./test_correctness_2 >> ./logs/test_correctness_2.log 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_3 test_correctness_3.cu
./test_correctness_3 >> ./logs/test_correctness_3.log 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_4 test_correctness_4.cu
./test_correctness_4 >> ./logs/test_correctness_4.log 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_5 test_correctness_5.cu
./test_correctness_5 >> ./logs/test_correctness_5.log 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_6 test_correctness_6.cu
./test_correctness_6 >> ./logs/test_correctness_6.log 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_7 test_correctness_7.cu
./test_correctness_7 >> ./logs/test_correctness_7.log 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_8 test_correctness_8.cu
./test_correctness_8 >> ./logs/test_correctness_8.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_1.cu -o benchmark_1 -lcurand
./benchmark_1 0 >> ./logs/benchmark_1_INPUT_MODE_0.log 2>&1
./benchmark_1 1 >> ./logs/benchmark_1_INPUT_MODE_1.log 2>&1
./benchmark_1 2 >> ./logs/benchmark_1_INPUT_MODE_2.log 2>&1
./benchmark_1 3 >> ./logs/benchmark_1_INPUT_MODE_3.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_2.cu -o benchmark_2 -lcurand
./benchmark_2 0 >> ./logs/benchmark_2_INPUT_MODE_0.log 2>&1
./benchmark_2 1 >> ./logs/benchmark_2_INPUT_MODE_1.log 2>&1
./benchmark_2 2 >> ./logs/benchmark_2_INPUT_MODE_2.log 2>&1
./benchmark_2 3 >> ./logs/benchmark_2_INPUT_MODE_3.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_3.cu -o benchmark_3 -lcurand
./benchmark_3 0 >> ./logs/benchmark_3_INPUT_MODE_0.log 2>&1
./benchmark_3 1 >> ./logs/benchmark_3_INPUT_MODE_1.log 2>&1
./benchmark_3 2 >> ./logs/benchmark_3_INPUT_MODE_2.log 2>&1
./benchmark_3 3 >> ./logs/benchmark_3_INPUT_MODE_3.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_4.cu -o benchmark_4 -lcurand
./benchmark_4 0 >> ./logs/benchmark_4_INPUT_MODE_0.log 2>&1
./benchmark_4 1 >> ./logs/benchmark_4_INPUT_MODE_1.log 2>&1
./benchmark_4 2 >> ./logs/benchmark_4_INPUT_MODE_2.log 2>&1
./benchmark_4 3 >> ./logs/benchmark_4_INPUT_MODE_3.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_5.cu -o benchmark_5 -lcurand
./benchmark_5 0 >> ./logs/benchmark_5_INPUT_MODE_0.log 2>&1
./benchmark_5 1 >> ./logs/benchmark_5_INPUT_MODE_1.log 2>&1
./benchmark_5 2 >> ./logs/benchmark_5_INPUT_MODE_2.log 2>&1
./benchmark_5 3 >> ./logs/benchmark_5_INPUT_MODE_3.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_6.cu -o benchmark_6 -lcurand
./benchmark_6 0 >> ./logs/benchmark_6_INPUT_MODE_0.log 2>&1
./benchmark_6 1 >> ./logs/benchmark_6_INPUT_MODE_1.log 2>&1
./benchmark_6 2 >> ./logs/benchmark_6_INPUT_MODE_2.log 2>&1
./benchmark_6 3 >> ./logs/benchmark_6_INPUT_MODE_3.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_7.cu -o benchmark_7 -lcurand
./benchmark_7 0 >> ./logs/benchmark_7_INPUT_MODE_0.log 2>&1
./benchmark_7 1 >> ./logs/benchmark_7_INPUT_MODE_1.log 2>&1
./benchmark_7 2 >> ./logs/benchmark_7_INPUT_MODE_2.log 2>&1
./benchmark_7 3 >> ./logs/benchmark_7_INPUT_MODE_3.log 2>&1

nvcc -O3 -arch=sm_90 benchmark_8.cu -o benchmark_8 -lcurand
./benchmark_8 0 >> ./logs/benchmark_8_INPUT_MODE_0.log 2>&1
./benchmark_8 1 >> ./logs/benchmark_8_INPUT_MODE_1.log 2>&1
./benchmark_8 2 >> ./logs/benchmark_8_INPUT_MODE_2.log 2>&1
./benchmark_8 3 >> ./logs/benchmark_8_INPUT_MODE_3.log 2>&1

cuobjdump --dump-resource-usage benchmark_1 | c++filt >> ./resource-usage/benchmark_1.resource-usage.txt 2>&1
cuobjdump --dump-resource-usage benchmark_2 | c++filt >> ./resource-usage/benchmark_2.resource-usage.txt 2>&1
cuobjdump --dump-resource-usage benchmark_3 | c++filt >> ./resource-usage/benchmark_3.resource-usage.txt 2>&1
cuobjdump --dump-resource-usage benchmark_4 | c++filt >> ./resource-usage/benchmark_4.resource-usage.txt 2>&1
cuobjdump --dump-resource-usage benchmark_5 | c++filt >> ./resource-usage/benchmark_5.resource-usage.txt 2>&1
cuobjdump --dump-resource-usage benchmark_6 | c++filt >> ./resource-usage/benchmark_6.resource-usage.txt 2>&1
cuobjdump --dump-resource-usage benchmark_7 | c++filt >> ./resource-usage/benchmark_7.resource-usage.txt 2>&1
cuobjdump --dump-resource-usage benchmark_8 | c++filt >> ./resource-usage/benchmark_8.resource-usage.txt 2>&1

nvcc -O3 -arch=sm_90 -o test_correctness_6.coarsened test_correctness_6.coarsened.cu
./test_correctness_6.coarsened >> ./logs/test_correctness_6.coarsened.log 2>&1