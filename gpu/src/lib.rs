use krnl::device::Device;
use matrix::OptimMatrix;
use warps::diamond_partitioning_gpu;

mod kernels;
mod matrix;
mod utils;
mod warps;

type FloatType = f32;

pub fn main1() {
    let x1 = (0..10000)
        .map(|_| rand::random::<FloatType>())
        .collect::<Vec<FloatType>>();
    let x2 = (0..10000)
        .map(|_| rand::random::<FloatType>())
        .collect::<Vec<FloatType>>();

    // assert_eq!(euclidean_distance(x1.clone(), x2.clone()), x1.iter().zip(x2.iter()).map(|(x1, x2)| (x1 - x2) * (x1 - x2)).sum::<FloatType>().sqrt());
    // println!("{}", euclidean_distance(x1, x2));
}

pub fn get_gpu_at_index(index: usize) -> krnl::device::Device {
    let devices: Vec<_> = [Device::builder().build().unwrap()]
            .into_iter()
            .chain((1..).map_while(|i| Device::builder().index(i).build().ok()))
            .collect();
        if devices.is_empty() {
            panic!("No device!");
        }
        let device = devices.iter().enumerate().map(|(i, x)|(i, x.info().unwrap())).max_by_key(|(_, x)| x.max_threads() * x.max_groups()).unwrap();

    krnl::device::Device::builder()
        .index(device.0)
        .build()
        .ok()
        .unwrap()
}

pub fn compute_test(device: krnl::device::Device, a: &[f64], b: &[f64]) -> f64 {
    let a = a.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let b = b.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    let res = diamond_partitioning_gpu(device, &a, &b, f32::INFINITY);
    res as f64
}
