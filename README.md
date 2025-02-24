# tsdistances

## Introduction

`tsdistances` is a Python library (with Rust backend) for computing various pairwise distances between sets of time series data. 

It provides eﬀicient implementation of elastic distance measures such as Dynamic Time Warping (DTW), Longest Common Subsequence (LCSS), Time Warping Edit (TWE), and many others.

The library is designed to be fast and scalable, leveraging parallel computation and GPU support via Vulkan for improved performance.

### Features

1.  Multiple Distance Measures: Supports a wide range of time series distance measures:

    -   Euclidean

    -   CATCH22 Euclidean

    -   Edit Distance with Real Penalty (ERP)

    -   Longest Common Subsequence (LCSS)

    -   Dynamic Time Warping (DTW)

    -   Derivative Dynamic Time Warping (DDTW)

    -   Weighted Dynamic Time Warping (WDTW)

    -   Weighted Derivative Dynamic Time Warping (WDDTW)

    -   Amerced Dynamic Time Warping (ADTW)

    -   Move-Split-Merge (MSM)

    -   Time Warp Edit Distance (TWE)

    -   Shape-Based Distance (SBD)

    -   MPDist

2.  Parallel Computation: Utilizes multiple CPU cores to speed up computations.

3.  GPU Acceleration: Optional GPU support using Vulkan for even faster computations.

## Installation

### Prerequisites {#prerequisites .unnumbered}

-   Python 3.10 or later

-   [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) (ver.
    1.3.283) for GPU functionalities

    -   [MacOS](https://sdk.lunarg.com/sdk/download/1.3.283.0/mac/vulkansdk-macos-1.3.283.0.dmg)

    -   [Linux]( https://sdk.lunarg.com/sdk/download/1.3.283.0/linux/vulkansdk-linux-x86_64-1.3.283.0.tar.xz)

    -   [Windows](https://sdk.lunarg.com/sdk/download/1.3.283.0/windows/VulkanSDK-1.3.283.0-Installer.exe)

Automatic GPU driver installation provided in
[Src/replicate_results.ipynb](Src/replicate_results.ipynb){.uri}.

### From Source

To install tsdistances from source, follow these steps:
```bash
    $ git clone https://github.com/albertoazzari/tsdistances/
    $ cd tsdistances
    $ python -m venv .venv
    $ source .venv/bin/activate
    $ pip install maturin
    $ maturin develop --release
```
### PIP

If you use pip, you can install tsdistances with:
```bash
    $ pip install tsdistances
```
## Usage

Here's a basic example of using tsdistances to compute the Dynamic Time
Warping (DTW) distance between two set of time series:
```python
        
    import numpy as np
    import tsdistances
    # Example usage of computing DTW distance
    x1 = np.array([
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]
                ])
    x2 = np.array([
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0]
                ])
    # Compute DTW distance
    result = tsdistances.dtw_distance(x1, x2, n_jobs=4, device='cpu')
    print(result)


```


