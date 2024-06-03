# tsdistances

`tsdistances` is a Python library (with Rust backend) for computing various pairwise distances between sets of time series data. It provides efficient implementations of elastic distance measures such as Dynamic Time Warping (DTW), Longest Common Subsequence (LCSS), and Time Warping Edit (TWE). The library is designed to be fast and scalable, leveraging parallel computation for improved performance.

## Installation

You can install `tsdistances` from source:

```bash
$ git clone https://github.com/albertoazzari/tsdistances/
$ cd tsdistances
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install maturin
$ maturin develop --release
```

## Usage
```python
import tsdistances

# Example usage of computing DTW distance
x1 = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]

x2 = [
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
]

# Compute DTW distance
result = tsdistances.dtw(x1, x2, band=1.0, cached=False, n_jobs=4)
print(result)
```



