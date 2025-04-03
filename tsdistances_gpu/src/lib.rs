#![deny(warnings)]
use kernels::warp::erp_distance::ERPImpl;
use std::{cell::OnceCell, sync::Arc};
use vulkano::{
    VulkanLibrary,
    device::{
        DeviceExtensions, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
};
use warps::{GpuBatchMode, diamond_partitioning_gpu};

mod kernels;
mod tests;
mod utils;
mod warps;

pub fn get_device() -> (Arc<PhysicalDevice>, u32) {
    let cell = OnceCell::new();
    let instance = cell.get_or_init(|| {
        let library = VulkanLibrary::new().unwrap();
        Instance::new(library, InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        })
        .unwrap()
    });
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();
    (physical_device, queue_family_index)
}

pub fn erp<'a, M: GpuBatchMode>(
    device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    device_extensions: DeviceExtensions,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    gap_penalty: f64,
) -> M::ReturnType {
    diamond_partitioning_gpu::<_, M>(
        device,
        queue_family_index,
        device_extensions,
        ERPImpl {
            gap_penalty: gap_penalty as f32,
        },
        a,
        b,
        f32::INFINITY,
    )
}
