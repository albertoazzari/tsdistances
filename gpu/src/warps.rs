use krnl::{buffer::Buffer, device::Device};

use crate::{kernels::warp::GpuKernelImpl, utils::next_multiple_of_n};

pub fn diamond_partitioning_gpu<G: GpuKernelImpl>(
    device: Device,
    params: G,
    a: &[f32],
    b: &[f32],
    init_val: f32,
) -> f32 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    let max_subgroup_threads: usize = device.info().unwrap().max_subgroup_threads() as usize;
    let new_a_len = next_multiple_of_n(a.len(), max_subgroup_threads);
    let new_b_len = next_multiple_of_n(b.len(), max_subgroup_threads);

    let mut a_padded = vec![0.0; new_a_len];
    let mut b_padded = vec![0.0; new_b_len];

    a_padded[..a.len()].copy_from_slice(a);
    b_padded[..b.len()].copy_from_slice(b);

    diamond_partitioning_gpu_::<G>(
        device,
        params,
        max_subgroup_threads,
        a.len(),
        b.len(),
        a_padded,
        b_padded,
        init_val,
    )
}

pub fn diamond_partitioning_gpu_<G: GpuKernelImpl>(
    device: Device,
    params: G,
    max_subgroup_threads: usize,
    a_len: usize,
    b_len: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    init_val: f32,
) -> f32 {
    let a_gpu = Buffer::from(a.clone()).into_device(device.clone()).unwrap();
    let b_gpu = Buffer::from(b.clone()).into_device(device.clone()).unwrap();

    let padded_a_len = next_multiple_of_n(a_len, max_subgroup_threads);
    let padded_b_len = next_multiple_of_n(b_len, max_subgroup_threads);

    let diag_len = 2 * (padded_a_len + 1).next_power_of_two();

    let mut diagonal = vec![init_val; diag_len];
    diagonal[0] = 0.0;

    let mut diagonal = Buffer::from(diagonal).into_device(device.clone()).unwrap();

    let a_diamonds = padded_a_len.div_ceil(max_subgroup_threads);
    let b_diamonds = padded_b_len.div_ceil(max_subgroup_threads);
    let rows_count = (padded_a_len + padded_b_len).div_ceil(max_subgroup_threads) - 1;

    let mut diamonds_count = 1;
    let mut first_coord = -(max_subgroup_threads as isize);
    let mut a_start = 0;
    let mut b_start = 0;

    // Number of kernel calls
    for i in 0..rows_count {
        params.dispatch(
            device.clone(),
            first_coord as i64,
            i as u64,
            diamonds_count as u64,
            a_start as u64,
            b_start as u64,
            a_len as u64,
            b_len as u64,
            max_subgroup_threads as u64,
            a_gpu.as_slice(),
            b_gpu.as_slice(),
            diagonal.as_slice_mut(),
        );

        if i < (a_diamonds - 1) {
            diamonds_count += 1;
            first_coord -= max_subgroup_threads as isize;
            a_start += max_subgroup_threads;
        } else if i < (b_diamonds - 1) {
            first_coord += max_subgroup_threads as isize;
            b_start += max_subgroup_threads;
        } else {
            diamonds_count -= 1;
            first_coord += max_subgroup_threads as isize;
            b_start += max_subgroup_threads;
        }
    }

    let diagonal = diagonal.into_vec().unwrap();

    fn index_mat_to_diag(i: usize, j: usize) -> (usize, isize) {
        (i + j, (j as isize) - (i as isize))
    }

    let (_, cx) = index_mat_to_diag(a_len, b_len);

    let res = diagonal[(cx as usize) & (diagonal.len() - 1)];

    res
}

// {
//     // Single kernel call
//     for j in 0..diamonds_count {
//         let diag_start = first_coord + ((j * max_subgroup_threads) as isize) * 2;
//         let d_a_start = a_start - j * max_subgroup_threads;
//         let d_b_start = b_start + j * max_subgroup_threads;

//         let alen = a_len - d_a_start;
//         let blen = b_len - d_b_start;

//         let a = &a;
//         let b = &b;

//         // Single warp
//         warp_kernel(
//             &mut matrix,
//             i * max_subgroup_threads,
//             d_a_start,
//             d_b_start,
//             diag_start + (max_subgroup_threads as isize),
//             (max_subgroup_threads * 2 + 1).min(alen + blen + 1),
//             |i, j, x, y, z| {
//                 let dist = (a[i] - b[j]).abs();
//                 dist + z.min(x.min(y))
//             },
//         );
//     }
// }

// pub fn warp_kernel<M: DiagonalMatrix>(
//     matrix: &mut M,
//     d_offset: usize,
//     a_start: usize,
//     b_start: usize,
//     diag_mid: isize,
//     diag_count: usize,
//     max_subgroup_threads: usize,
//     dist_lambda: impl Fn(usize, usize, f32, f32, f32) -> f32,
// ) {
//     let mut i = a_start;
//     let mut j = b_start;
//     let mut s = diag_mid;
//     let mut e = diag_mid;

//     for d in 2..diag_count {
//         for warp in 0..32 {
//             let k = (warp * 2) as isize + s;

//             if k <= e {
//                 let i1 = i - warp;
//                 let j1 = j + warp;

//                 let dleft = matrix.get_diagonal_cell(d_offset + d - 1, k - 1);
//                 let ddiag = matrix.get_diagonal_cell(d_offset + d - 2, k);
//                 let dup = matrix.get_diagonal_cell(d_offset + d - 1, k + 1);

//                 let value = dist_lambda(i1, j1, dleft, ddiag, dup);

//                 matrix.set_diagonal_cell(d_offset + d, k, value);
//             }
//         }
//         // Warp synchronize

//         if d <= max_subgroup_threads {
//             i += 1;
//             s -= 1;
//             e += 1;
//         } else {
//             j += 1;
//             s += 1;
//             e -= 1;
//         }
//     }
// }
