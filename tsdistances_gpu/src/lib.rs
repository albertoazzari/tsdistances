#![cfg_attr(target_arch = "spirv", no_std)]
#![deny(warnings)]
use spirv_std::{glam, spirv};
use spirv_std::num_traits::Float;

pub fn euclidean_distance(a: &i32, b: &i32) -> f32 {
    ((a - b) * (a - b)) as f32
}

#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input_a: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_b: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let index = id.x as usize;
    
    // Only the first thread calculates the total distance
    if index == 0 {
        // Calculate total squared distance
        let mut total_distance = 0.0f32;
        for i in 0..input_a.len() {
            total_distance += euclidean_distance(&input_a[i], &input_b[i]);
        }
        
        // Take square root of total distance
        output[0] = total_distance.sqrt();
    }
}