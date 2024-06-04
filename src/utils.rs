use std::mem::transmute;

#[derive(Clone)]
pub struct FloatVecEq(pub Vec<f64>);

impl PartialEq for FloatVecEq {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let self_: &Vec<FloatEq> = transmute(&self.0);
            let other_: &Vec<FloatEq> = transmute(&other.0);

            self_.eq(other_)
        }
    }
}

impl Eq for FloatVecEq {}

impl std::hash::Hash for FloatVecEq {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            let self_: &Vec<FloatEq> = transmute(&self.0);
            self_.hash(state);
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FloatEq(pub f64);

impl PartialEq for FloatEq {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for FloatEq {}

impl std::hash::Hash for FloatEq {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl std::cmp::PartialOrd for FloatEq {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl std::cmp::Ord for FloatEq {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

pub fn check_sakoe_chiba_band(band: f64) -> Result<(), pyo3::PyErr> {
    if band < 0.0 || band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Sakoe-Chiba band radius must be less than the length of the timeseries"));
    }

    Ok(())
}