use kernels::warp::{DtwImpl, ErpImpl};
use warps::diamond_partitioning_gpu;

pub mod device;
mod kernels;
mod utils;
mod warps;

type FloatType = f32;

#[test]
fn list_gpus() {
    let devices: Vec<_> = [Device::builder().build().unwrap()]
        .into_iter()
        .chain((1..).map_while(|i| Device::builder().index(i).build().ok()))
        .collect();

    for device in devices {
        println!("{:#?}", device.info());
    }
}

pub fn compute_test_dtw(device: krnl::device::Device, a: &[f64], b: &[f64]) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let res = diamond_partitioning_gpu(device, DtwImpl {}, &a, &b, f32::INFINITY);
    res as f64
}

pub fn compute_test_erp(device: krnl::device::Device, a: &[f64], b: &[f64]) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let res = diamond_partitioning_gpu(device, ErpImpl { gap_penalty: 7.0 }, &a, &b, f32::INFINITY);
    res as f64
}
