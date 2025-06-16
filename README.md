# tsdistances

## Introduction

`tsdistances` is a Python library (with Rust backend) for computing various pairwise distances between sets of time series data. 

It provides eﬀicient implementation of elastic distance measures such as Dynamic Time Warping (DTW), Longest Common Subsequence (LCSS), Time Warping Edit (TWE), and many others.

The library is designed to be fast and scalable, leveraging parallel computation and GPU support via Vulkan for improved performance.

### Features

1.  Multiple Distance Measures: Supports a wide range of time series distance measures:

    -   Euclidean

    -   CATCH22 Euclidean

    -   Edit Distance with Real Penalty (ERP) optionally with GPU support

    -   Longest Common Subsequence (LCSS) optionally with GPU support

    -   Dynamic Time Warping (DTW) optionally with GPU support

    -   Derivative Dynamic Time Warping (DDTW) optionally with GPU support

    -   Weighted Dynamic Time Warping (WDTW) optionally with GPU support

    -   Weighted Derivative Dynamic Time Warping (WDDTW) optionally with GPU support

    -   Amerced Dynamic Time Warping (ADTW) optionally with GPU support

    -   Move-Split-Merge (MSM) optionally with GPU support

    -   Time Warp Edit Distance (TWE) optionally with GPU support

    -   Shape-Based Distance (SBD)

    -   MPDist

2.  Parallel Computation: Utilizes multiple CPU cores to speed up computations.

3.  GPU Acceleration: Optional GPU support based on [Vulkan](https://www.vulkan.org/) for even faster computations with [Rust-GPU](https://rust-gpu.github.io/).

## Benchmark

To evaluate the performance of our time series distance computation library, we conducted a comparative analysis with existing libraries. 

We selected [AEON](https://github.com/aeon-toolkit/aeon) as the primary competitor due to its comprehensive implementation of distance metrics, making it the most suitable for direct comparison. Several other libraries were considered, and while we did not conduct a full benchmark on all datasets, we reported their execution times on a subset of the [UCR Archive datasets](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

| Dataset                              | Aeon      | tsdistances | tsdistances PAR | tsdistances GPU |
|--------------------------------------|-----------|-------------|-----------------|-----------------|
|ACSF1                       | 8854.71   | 786.84      | 131.36          | 10.10           |
|Adiac                       | 1811.02   | 181.84      | 14.42           | 6.60            |
|Beef                        | 77.04     | 7.44        | 3.77            | 0.34            |
|CBF                         | 88.22     | 9.06        | 0.85            | 0.60            |
|ChlorineConcentration       | 10674.45  | 1070.61     | 74.84           | 41.98           |
|CinCECGTorso                | 30180.62  | 2807.16     | 229.49          | 28.03           |
|CricketX                    | 5252.95   | 516.66      | 40.55           | 13.63           |
|DiatomSizeReduction         | 117.24    | 11.80       | 1.68            | 0.43            |
|DistalPhalanxOutlineCorrect | 654.23    | 68.06       | 6.41            | 5.34            |
|ECG200                      | 35.73     | 3.89        | 0.72            | 0.33            |
|EthanolLevel                | 311557.03 | 28509.74    | 2033.03         | 283.55          |
|FreezerRegularTrain         | 7853.39   | 768.72      | 57.35           | 20.68           |
|FreezerSmallTrain           | 1405.80   | 138.01      | 10.48           | 4.53            |
|Ham                         | 837.90    | 81.00       | 12.60           | 1.91            |
|Haptics                     | 16487.36  | 1579.31     | 179.22          | 22.78           |
|HouseTwenty                 | 4926.42   | 467.69      | 95.01           | 4.81            |
|ItalyPowerDemand            | 9.99      | 1.12        | 0.14            | 0.29            |
|MixedShapesSmallTrain       | 50890.02  | 4877.02     | 358.87          | 74.37           |
|NonInvasiveFetalECGThorax1  | 733747.73 | 70415.83    | 5005.24         | 811.66          |
|ShapesAll                   | 38197.35  | 3494.29     | 301.04          | 67.61           |
|Strawberry                  | 6383.30   | 628.60      | 52.32           | 22.43           |
|UWaveGestureLibraryX        | 76309.38  | 7487.65     | 524.81          | 155.80          |
|Wafer                       | 31792.18  | 3217.30     | 229.56          | 126.54          |

 Sum of the times (in seconds) taken by each library for each dataset to compute all the distances (Euclidean, ERP, LCSS, DTW, ..., SBD). Excluded from the benchmark `CATCH22 Euclidean` and `MPDist` which were implemented later.  CPU: Intel(R) Core(TM) i9-10980HK GPU: NVIDIA GeForce RTX 4090

## Installation
### PIP

If you use pip, you can install tsdistances with:
```bash
    $ pip install tsdistances
```

### From Source

This can be done by going through the following steps in sequence:
1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
2. Install [maturin](https://maturin.rs/): `pip install maturin`
3. `maturin develop --release` to build the library, if want to build also the gpu part:
    a. Install [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
    b. Either install [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools.git) or compile `tsdistances_gpu` with `cargo build --release --no-default-features --use-compiled-tools` 

## Usage

### Example 1: Compute DTW Distance on CPU and GPU
```python
        
    import numpy as np
    import tsdistances

    # Generate two random time series (1-D arrays of length 100)
    np.random.seed(0)
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)

    # Compute DTW distance on CPU
    cpu_distance = tsdistances.dtw_distance(x1, x2, device='cpu')
    print(f"DTW distance (CPU): {cpu_distance}")

    gpu_distance = tsdistances.dtw_distance(x1, x2, device='gpu')

    print(f"DTW distance (GPU): {gpu_distance}")
```

### Example 2: Pairwise Distances with Multiple Time Series and Parallel Computation
```python
    import numpy as np
    import tsdistances

    # Generate a batch of 10 random time series (each of length 50)
    np.random.seed(42)
    X = np.random.rand(10, 50)

    # Pairwise DTW distances within the set X (on CPU, single thread)
    pairwise_distances = tsdistances.dtw_distance(X, par=False, device='cpu')
    print("Pairwise DTW distance matrix (CPU, 4 jobs):")
    print(pairwise_distances)

    # Compare two batches: compute distances between each element of X and each element of Y
    Y = np.random.rand(8, 50)
    batch_distances = tsdistances.dtw_distance(X, Y, par=True, device='cpu')
    print("Batch DTW distance matrix (X vs Y):")
    print(batch_distances)
```
Notes
1. `device='gpu'` enables GPU acceleration.

2. `par` controls parallelism. Set it to `True` to use all available CPU cores.

3. If `v` is not provided, the function computes pairwise distances within `u`.

Important: Results will differ between CPU and GPU due to floating-point precision:

    CPU computations use float64 (double precision) for higher numerical accuracy.

    GPU computations use float32 (single precision) for better performance.
    For instance, on an RTX 4090:

        FP32 performance: 82.58 TFLOPS

        FP64 performance: 1.29 TFLOPS (1:64 rate)
        Using float32 on GPU drastically improves speed but introduces small numerical differences compared to CPU results.

## Testing and Validation

All distance implementations in `tsdistances` are **tested against [AEON](https://github.com/aeon-toolkit/aeon)**, a widely-used Python library for time series analysis and distances. This ensures that the results are correct and consistent with established benchmarks in the field.

To run the correctness tests, simply use `pytest`:

```bash
pytest -v tests/test_correctness_cpu.py
```

