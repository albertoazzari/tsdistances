pub mod diagonal;
mod distances;
mod utils;
pub mod warps;
pub mod matrix;

use ctrlc;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "tsdistances")]
fn py_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let _ = ctrlc::set_handler(move || {
        println!("\nraise KeyboardInterrupt (Ctrl+C pressed)");
        std::process::exit(1);
    });
    m.add_function(wrap_pyfunction!(distances::euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(distances::erp, m)?)?;
    m.add_function(wrap_pyfunction!(distances::lcss, m)?)?;
    m.add_function(wrap_pyfunction!(distances::dtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::ddtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::wdtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::wddtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::adtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::msm, m)?)?;
    m.add_function(wrap_pyfunction!(distances::twe, m)?)?;
    m.add_function(wrap_pyfunction!(distances::sbd, m)?)?;
    Ok(())
}
