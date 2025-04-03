#[cfg(test)]
mod tests {

    #[test]
    fn test_device() {
        use crate::get_device;
        let (physical_device, queue_family_index) = get_device();
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );
        println!("Queue family index: {}", queue_family_index);
    }
}
