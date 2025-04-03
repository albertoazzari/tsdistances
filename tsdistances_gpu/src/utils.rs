use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    device::Device,
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryTypeFilter, StandardMemoryAllocator,
    },
};

pub fn move_to<T: BufferContents + Copy>(data: &[T], device: Arc<Device>) -> Subbuffer<[T]> {
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let buffer_device = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
            ..Default::default()
        },
        data.len() as u64,
    )
    .unwrap_or_else(|e| {
        panic!("Failed to create buffer of len {}\n {:?}", data.len(), e);
    });
    buffer_device
}
