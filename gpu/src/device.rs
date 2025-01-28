use dashmap::DashMap;
use krnl::device::Device;
use lazy_static::lazy_static;
use vulkano::{
    device::{physical::PhysicalDeviceType, DeviceExtensions, QueueFlags},
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

pub fn find_best_device_index() -> usize {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let gpu_index = instance
        .enumerate_physical_devices()
        .unwrap()
        .enumerate()
        .filter(|(_, p)| p.supported_extensions().contains(&device_extensions))
        .filter_map(|(i, p)| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one
            // queue that supports compute operations.
            Some((
                i,
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                    .map(|i| (p, i as u32))?,
            ))
        })
        .min_by_key(|(_, (p, _))| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap()
        .0;
    gpu_index
}

lazy_static! {
    static ref GPUS_MAP: DashMap<usize, Device> = DashMap::new();
}


pub fn get_best_gpu() -> krnl::device::Device {
    get_gpu_at_index(find_best_device_index())
}

pub fn get_gpu_at_index(index: usize) -> krnl::device::Device {
    GPUS_MAP.entry(index).or_insert_with(|| {
        krnl::device::Device::builder()
            .index(index)
            .build()
            .ok()
            .unwrap()
    }).clone()
}
