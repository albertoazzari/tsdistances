use kernels::warp::{ADTWImpl, DTWImpl, ERPImpl, LCSSImpl, MSMImpl, TWEImpl, WDTWImpl};
use warps::diamond_partitioning_gpu;

mod kernels;
mod warps;
pub mod device;

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

pub fn erp(device: krnl::device::Device, a: &[f64], b: &[f64], gap_penalty: f64) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let res = diamond_partitioning_gpu(device, ERPImpl { gap_penalty: gap_penalty as f32 }, &a, &b, f32::INFINITY);
    res as f64
}

pub fn lcss(device: krnl::device::Device, a: &[f64], b: &[f64], epsilon: f64) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let res = diamond_partitioning_gpu(device, LCSSImpl { epsilon: epsilon as f32 }, &a, &b, 0.0);
    res as f64
}

pub fn dtw(device: krnl::device::Device, a: &[f64], b: &[f64]) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let res = diamond_partitioning_gpu(device, DTWImpl {}, &a, &b, f32::INFINITY);
    res as f64
}

pub fn wdtw(device: krnl::device::Device, a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let weights = weights.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    
    let res = diamond_partitioning_gpu(device.clone(), WDTWImpl {weights: krnl::buffer::Buffer::from(weights).into_device(device).unwrap()}, &a, &b, f32::INFINITY);
    res as f64
}

pub fn msm(device: krnl::device::Device, a: &[f64], b: &[f64]) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let res = diamond_partitioning_gpu(device, MSMImpl {}, &a, &b, f32::INFINITY);
    res as f64
}

pub fn twe(device: krnl::device::Device, a: &[f64], b: &[f64], stiffness: f64, penalty: f64) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    
    let res = diamond_partitioning_gpu(device, TWEImpl {stiffness: stiffness as f32, penalty: penalty as f32}, &a, &b, f32::INFINITY);
    res as f64
}

pub fn adtw(device: krnl::device::Device, a: &[f64], b: &[f64], w: f64) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    
    let res = diamond_partitioning_gpu(device, ADTWImpl {w: w as f32}, &a, &b, f32::INFINITY);
    res as f64
}
