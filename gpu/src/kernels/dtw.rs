use krnl::{
    anyhow::Result,
    buffer::{Buffer, Slice, SliceMut},
    device::Device,
    macros::module,
};

#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    pub fn dtw_impl(x1: f32, x2: f32, y: &mut f32) {
        *y += (x1 - x2) * (x1 - x2);
    }

    #[kernel]
    pub fn dtw(#[item] x1: f32, #[item] x2: f32, #[item] y: &mut f32) {
        dtw_impl(x1, x2, y);
    }

    #[kernel]
    pub fn dtw_global(
        #[global] x1: Slice<f32>,
        #[global] x2: Slice<f32>,
        #[global] y: UnsafeSlice<f32>,
    ) {
        use krnl_core::buffer::UnsafeIndex;

        let global_id = kernel.global_id();
        if global_id < x1.len().min(x2.len()).min(y.len()) {
            dtw_impl(x1[global_id], x2[global_id], unsafe {
                y.unsafe_index_mut(global_id)
            });
        }
    }
}

fn dtw(x1: Slice<f32>, x2: Slice<f32>, mut y: SliceMut<f32>) -> Result<()> {
    if true {
        // use local threads
        println!("local");
        kernels::dtw::builder()?
            .build(y.device())?
            .dispatch(x1, x2, y);
    } else {
        // or
        println!("global");
        kernels::dtw_global::builder()?
            .build(y.device())?
            .with_global_threads(y.len() as u32)
            .dispatch(x1, x2, y);
    }
    Ok(())
}

pub fn dtw_distance(x1: Vec<f32>, x2: Vec<f32>) -> f32 {
    let y = vec![0f32; x1.len()];
    let device = Device::builder().build().ok().unwrap_or(Device::host());
    let x1_in_device = Buffer::from(x1).into_device(device.clone()).unwrap();
    let x2_in_device = Buffer::from(x2).into_device(device.clone()).unwrap();
    let mut y_in_device = Buffer::from(y).into_device(device.clone()).unwrap();
    dtw(
        x1_in_device.as_slice(),
        x2_in_device.as_slice(),
        y_in_device.as_slice_mut(),
    )
    .unwrap();
    let y = y_in_device.into_vec().unwrap();
    y.iter().sum::<f32>().sqrt()
}
