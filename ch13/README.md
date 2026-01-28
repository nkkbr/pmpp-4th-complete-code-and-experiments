# Chapter 13: Sorting

[简体中文](https://github.com/nkkbr/pmpp/blob/main/ch13/README.zh-CN.md)

This chapter primarily introduces the implementation of sorting algorithms. Although the original book covers various sorting methods such as Merge Sort, the experiments in this project focus on the CUDA implementation and performance optimization of **Radix Sort**.

## Kernels and Experiments

The experimental projects included in this chapter are shown in the table below:

| Experiment ID | Kernel File | Correctness Verification | Performance Benchmark | Improvement Baseline |
| --- | --- | --- | --- | --- |
| **Experiment 1** | `sort_kernels_1.cuh` | `test_correctness_1.cu` | `benchmark_1.cu` | - |
| **Experiment 2** | `sort_kernels_2.cuh` | `test_correctness_2.cu` | `benchmark_2.cu` | Experiment 1 |
| **Experiment 3** | `sort_kernels_3.cuh` | `test_correctness_3.cu` | `benchmark_3.cu` | Experiment 2 |
| **Experiment 4** | `sort_kernels_4.cuh` | `test_correctness_4.cu` | `benchmark_4.cu` | Experiment 3 |
| **Experiment 5** | `sort_kernels_5.cuh` | `test_correctness_5.cu` | `benchmark_5.cu` | Experiment 3 |
| **Experiment 6** | `sort_kernels_6.cuh` | `test_correctness_6.cu` | `benchmark_6.cu` | Experiment 5 |
| **Experiment 7** | `sort_kernels_7.cuh` | `test_correctness_7.cu` | `benchmark_7.cu` | Experiment 4, Experiment 5 |
| **Experiment 8** | `sort_kernels_8.cuh` | `test_correctness_8.cu` | `benchmark_8.cu` | Experiment 6, Experiment 7 |
| **Experiment 9** | `sort_kernels_6.coarsened.cuh` | `test_correctness_6.coarsened.cu` | `benchmark_6.coarsened.cu` | Experiment 6 |

### Experiment Details

#### Experiment 1: Naive Implementation

This is the most naive baseline version. This version directly reads/writes Global Memory throughout, with the lowest expected performance.

* **Note**: The pseudo-code provided in Figure 13.4 contains `exclusiveScan(bits, N)`. This is a global Scan operation, not an intra-Block Scan. Therefore, it requires launching a new Kernel. To focus on Radix Sort itself, we directly called `thrust::exclusive_scan` in the intermediate steps.

#### Experiment 2: Shared Memory Optimization

Introduces Shared Memory. Each Block first sorts internally within Shared Memory, then performs coalesced access to write to Global Memory.

#### Experiment 3: Radix-4

Expands the radix from 2 (1-bit) to 4 (2-bit).

#### Experiment 4: Radix-8 (Basic)

Based on Experiment 3, further expands the radix to 8 (3-bit).

#### Experiment 5: Optimized Counting

Based on `sort_kernels_3.cuh`, optimized `radix_sort_count_block`.

* **Optimization Point**: Utilizes Warp-level primitives. Compared to `__shfl_down_sync`, which is suitable for collecting specific data, here we used `__ballot_sync` combined with `__popc`, which is more suitable for collecting Bool states, to efficiently count the number of "1"s.

#### Experiment 6: Optimized Scattering

Based on `sort_kernels_5.cuh`, optimized `radix_sort_coalesced_scatter`.

* **Optimization Point**: Replaced the low-performance version of the `Kogge-Stone algorithm` based on `Chapter 11` from the original book, switching to an efficient implementation using `__ballot_sync` and `__popc`.

#### Experiment 7: Radix-8 + Optimized Counting

Based on the optimization logic of `sort_kernels_5.cuh`, expanding the radix to 8.

#### Experiment 8: Radix-8 + Optimized Scattering

Based on the optimization logic of `sort_kernels_6.cuh`, expanding the radix to 8.

#### Experiment 9: Coarsening (Thread Coarsening)

Selected the previously best-performing Experiment 6 Kernel (`sort_kernels_6.cuh`) for coarsening treatment. That is, each Thread processes multiple data elements.

* We set adjustable parameters `BLOCK_SIZE` and `COARSEN` (coarsening level) to perform a parameter space search.

---

## Experimental Results and Analysis

### 1. Performance Evolution Analysis of Experiments 1-8

**Test Environment**: NVIDIA H200 (141GB)
**Test Sequences**: All-zero sequence, ascending sequence, descending sequence, random sequence.

**Key Observations:**

1. **Shared Memory Overhead (Exp 2 vs Exp 1)**:
Experiment 2 performed worse than Experiment 1. This indicates that the increased algorithmic complexity introduced by Shared Memory incurred overhead far exceeding its memory access benefits. The powerful L2 Cache of the H200 makes the naive version directly accessing Global Memory still competitive.
* *Note*: The `Kogge-Stone` algorithm implementation for Scan used in the code (derived from Chapter 11) had excessively low performance, mainly serving only algorithmic educational purposes, severely dragging down performance.

2. **Radix Size Trade-off (Exp 3 vs Exp 4)**:
Experiment 3 outperformed Experiment 2, indicating that increasing the Radix reduced the number of iterations and improved efficiency. However, Experiment 4 performed worse than Experiment 3 because Radix-8 caused a significant increase in register and Shared Memory usage, thereby reducing Occupancy.
3. **Advantage of Warp-level Primitives (Exp 6)**:
Experiment 6 performed the best. This confirms the high efficiency of prioritizing data collection and processing within the Warp. Utilizing registers for Warp-level operations reduced reliance on Shared Memory.
4. **Resource Bottleneck (Exp 7 & 8)**:
Although Experiments 7 and 8 adopted optimization strategies, the high resource consumption brought by Radix-8 led to a decline in Occupancy, and performance was still inferior to the optimized versions of Radix-4 (Experiments 5 and 6).

![Radix Sort Benchmark](./figures/performance_heatmap.png)
![resource-usage](./figures/resource_usage_analysis.png)

### 2. Experiment 9: Parameter Space Search (Performance Heatmap)

We conducted an exhaustive test on pairwise combinations of `Block_Size` (32~1024) and `COARSEN` (1~1024).

* **Note**: `-` in the table indicates that the combination could not launch the Kernel due to excessive register or Shared Memory resource requirements.

#### 2.1 Kernel Runtime (ms)

The data shows a clear performance "Sweet Spot".

| Block Size \ Coarsening |          1 |             2 |             4 |             8 |            16 |        32 |        64 |       128 |        256 |
| ----------------------: | ---------: | ------------: | ------------: | ------------: | ------------: | --------: | --------: | --------: | ---------: |
|                  **32** | 161.592827 |     80.971284 |     45.418840 |     33.526192 | **28.319040** | 45.792926 | 46.436958 | 81.283205 | 150.499213 |
|                  **64** |  80.936179 |     40.702309 |     29.813076 | **27.708961** |     30.541148 | 46.336899 | 47.622131 | 84.207284 |          - |
|                 **128** |  44.165486 |     29.711831 | **26.044274** |     28.015491 |     32.274519 | 49.250863 | 50.157660 |         - |          - |
|                 **256** |  38.975468 |     29.266344 | **28.333358** |     29.551520 |     41.149399 | 55.531062 |         - |         - |          - |
|                 **512** |  47.721560 | **35.593579** |     35.249997 |     42.485775 |     70.458574 |         - |         - |         - |          - |
|                **1024** |  66.418224 | **49.938743** |     64.013246 |     59.129605 |             - |         - |         - |         - |          - |

We created the following plots to visualize this set of data.

![1](./figures/1.png)
![2](./figures/2.png)
![3](./figures/3.png)
![4](./figures/4.png)

#### 2.2 Resource Usage Analysis

The following tables record the resource consumption of key Kernels under different configurations, explaining why some configurations failed to launch.

#### 1️⃣ `radix_sort_coalesced_scatter` —— REG

| Block Size \ Coarsening | 1  | 2  | 4  | 8  | 16 | 32  | 64 | 128 | 256 |
| ----------------------- | -- | -- | -- | -- | -- | --- | -- | --- | --- |
| 32                      | 19 | 27 | 36 | 51 | 64 | 98  | 32 | 32  | 32  |
| 64                      | 22 | 30 | 36 | 55 | 72 | 102 | 32 | 32  | -   |
| 128                     | 26 | 30 | 40 | 51 | 72 | 98  | 32 | -   | -   |
| 256                     | 26 | 30 | 36 | 48 | 72 | 102 | -  | -   | -   |
| 512                     | 30 | 30 | 40 | 47 | 72 | -   | -  | -   | -   |
| 1024                    | 26 | 30 | 36 | 47 | -  | -   | -  | -   | -   |

#### 2️⃣ `radix_sort_coalesced_scatter` —— SHARED（Bytes）

| Block Size \ Coarsening | 1    | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ----------------------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 32                      | 1204 | 1368  | 1696  | 2352  | 3664  | 6288  | 11536 | 22032 | 43024 |
| 64                      | 1368 | 1696  | 2352  | 3664  | 6288  | 11536 | 22032 | 43024 | -     |
| 128                     | 1696 | 2352  | 3664  | 6288  | 11536 | 22032 | 43024 | -     | -     |
| 256                     | 2352 | 3664  | 6288  | 11536 | 22032 | 43024 | -     | -     | -     |
| 512                     | 3664 | 6288  | 11536 | 22032 | 43024 | -     | -     | -     | -     |
| 1024                    | 6288 | 11536 | 22032 | 43024 | -     | -     | -     | -     | -     |

#### 3️⃣ `radix_sort_count_block` —— REG

| Block Size \ Coarsening | 1  | 2  | 4  | 8  | 16 | 32 | 64 | 128 | 256 |
| ----------------------- | -- | -- | -- | -- | -- | -- | -- | --- | --- |
| 32                      | 20 | 20 | 20 | 29 | 32 | 32 | 32 | 32  | 32  |
| 64                      | 20 | 20 | 20 | 29 | 32 | 32 | 32 | 32  | -   |
| 128                     | 22 | 22 | 22 | 29 | 32 | 32 | 32 | -   | -   |
| 256                     | 32 | 32 | 32 | 32 | 32 | 32 | -  | -   | -   |
| 512                     | 34 | 34 | 34 | 34 | 39 | -  | -  | -   | -   |
| 1024                    | 34 | 34 | 34 | 34 | -  | -  | -  | -   | -   |

#### 4️⃣ `radix_sort_count_block` —— SHARED（Bytes）

| Block Size \ Coarsening | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 256  |
| ----------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 32                      | 1036 | 1036 | 1036 | 1036 | 1036 | 1036 | 1036 | 1036 | 1036 |
| 64                      | 1048 | 1048 | 1048 | 1048 | 1048 | 1048 | 1048 | 1048 | -    |
| 128                     | 1072 | 1072 | 1072 | 1072 | 1072 | 1072 | 1072 | -    | -    |
| 256                     | 1120 | 1120 | 1120 | 1120 | 1120 | 1120 | -    | -    | -    |
| 512                     | 1216 | 1216 | 1216 | 1216 | 1216 | -    | -    | -    | -    |
| 1024                    | 1408 | 1408 | 1408 | 1408 | -    | -    | -    | -    | -    |

![5](./figures/5.png)

> **Conclusion**: The maximum Coarsening Factor that each Block Size can run corresponds exactly to the point where the Shared Memory usage of the `scatter` kernel approaches the hardware limit (approximately 43024 bytes).

## Acknowledgments

The experimental code implementation, data analysis, and chart production for this chapter, as well as this README file itself, were supported by **ChatGPT 5.2** and **Gemini 3 Pro**.