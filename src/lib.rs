mod distances;
mod elastic_distances;
mod utils;

use pyo3::prelude::*;


#[pymodule]
#[pyo3(name = "elastic_distances")]
fn py_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distances::euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(distances::dtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::ddtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::wdtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::wddtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::adtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::msm, m)?)?;
    m.add_function(wrap_pyfunction!(distances::twe, m)?)?;
    Ok(())
}