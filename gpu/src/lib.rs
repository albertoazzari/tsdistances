use std::error::Error;

use csv::ReaderBuilder;
use device::get_best_gpu;
use kernels::warp::{
    adtw_distance::ADTWImpl, dtw_distance::DTWImpl, erp_distance::ERPImpl, lcss_distance::LCSSImpl,
    msm_distance::MSMImpl, twe_distance::TWEImpl, wdtw_distance::WDTWImpl,
};
use warps::diamond_partitioning_gpu;

pub mod device;
mod kernels;
mod warps;

pub use warps::{GpuBatchMode, MultiBatchMode, SingleBatchMode};

fn read_csv<T>(file_path: &str) -> Result<Vec<Vec<T>>, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: 'static + Error, // needed to convert parsing error into Box<dyn Error>
{
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Vec<T> = record
            .iter()
            .map(|s| s.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        records.push(row);
    }
    Ok(records)
}

fn write_csv<T>(file_path: &str, data: &[Vec<T>]) -> Result<(), Box<dyn Error>>
where
    T: std::fmt::Display,
{
    let mut writer = csv::Writer::from_path(file_path)?;

    for row in data {
        writer.write_record(row.iter().map(|s| s.to_string()))?;
    }
    writer.flush()?;
    Ok(())
}

#[test]
pub fn test_erp() {
    let device = get_best_gpu();
    let data = read_csv("ts.csv").unwrap();
    let gap_penalty = 1.0;
    let start_time = std::time::Instant::now();
    let result = erp::<MultiBatchMode>(
        device.clone(),
        data.as_slice(),
        data.as_slice(),
        gap_penalty,
    );
    println!("Time taken: {:?}", start_time.elapsed());
    write_csv("erp_ts.csv", &result).unwrap();
}

#[test]
pub fn test_lcss() {
    let device = get_best_gpu();
    let data = read_csv("ts.csv").unwrap();
    let epsilon = 1.0;
    let start_time = std::time::Instant::now();
    let result = lcss::<MultiBatchMode>(device.clone(), data.as_slice(), data.as_slice(), epsilon);
    println!("Time taken: {:?}", start_time.elapsed());
    write_csv("lcss_ts.csv", &result).unwrap();
}

pub fn erp<'a, M: GpuBatchMode>(
    device: krnl::device::Device,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    gap_penalty: f64,
) -> M::ReturnType {
    diamond_partitioning_gpu::<_, M>(
        device,
        ERPImpl {
            gap_penalty: gap_penalty as f32,
        },
        a,
        b,
        f32::INFINITY,
    )
}

pub fn lcss<'a, M: GpuBatchMode>(
    device: krnl::device::Device,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    epsilon: f64,
) -> M::ReturnType {
    let similarity = diamond_partitioning_gpu::<_, M>(
        device,
        LCSSImpl {
            epsilon: epsilon as f32,
        },
        a,
        b,
        0.0,
    );
    let min_len = M::get_sample_length(&a.clone()).min(M::get_sample_length(&b.clone())) as f64;
    M::apply_fn(similarity, |s| 1.0 - s / min_len)
}

pub fn dtw<'a, M: GpuBatchMode>(
    device: krnl::device::Device,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
) -> M::ReturnType {
    diamond_partitioning_gpu::<_, M>(device, DTWImpl {}, a, b, f32::INFINITY)
}

pub fn wdtw<'a, M: GpuBatchMode>(
    device: krnl::device::Device,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    weights: &[f64],
) -> M::ReturnType {
    let weights = weights.iter().map(|x| *x as f32).collect::<Vec<f32>>();

    diamond_partitioning_gpu::<_, M>(
        device.clone(),
        WDTWImpl {
            weights: krnl::buffer::Buffer::from(weights)
                .into_device(device.clone())
                .unwrap(),
        },
        a,
        b,
        f32::INFINITY,
    )
}

pub fn msm<'a, M: GpuBatchMode>(
    device: krnl::device::Device,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
) -> M::ReturnType {
    diamond_partitioning_gpu::<_, M>(device, MSMImpl {}, a, b, f32::INFINITY)
}

pub fn twe<'a, M: GpuBatchMode>(
    device: krnl::device::Device,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    stiffness: f64,
    penalty: f64,
) -> M::ReturnType {
    diamond_partitioning_gpu::<_, M>(
        device,
        TWEImpl {
            stiffness: stiffness as f32,
            penalty: penalty as f32,
        },
        a,
        b,
        f32::INFINITY,
    )
}

pub fn adtw<'a, M: GpuBatchMode>(
    device: krnl::device::Device,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    w: f64,
) -> M::ReturnType {
    diamond_partitioning_gpu::<_, M>(device, ADTWImpl { w: w as f32 }, a, b, f32::INFINITY)
}
