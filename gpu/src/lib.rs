use kernels::warp::{
    adtw_distance::ADTWImpl, dtw_distance::DTWImpl, erp_distance::ERPImpl, lcss_distance::LCSSImpl,
    msm_distance::MSMImpl, twe_distance::TWEImpl, wdtw_distance::WDTWImpl,
};
use warps::diamond_partitioning_gpu;

pub mod device;
mod kernels;
mod warps;

pub use warps::{GpuBatchMode, MultiBatchMode, SingleBatchMode};

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
    diamond_partitioning_gpu::<_, M>(
        device,
        LCSSImpl {
            epsilon: epsilon as f32,
        },
        a,
        b,
        0.0,
    )
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
