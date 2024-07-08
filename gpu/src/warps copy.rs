use crate::{kernels::warp_gpu::kernels::DiagonalMatrix, utils::next_multiple_of_n};
const DIAMOND_SIZE: usize = 64;

#[test]
fn test_diamond_partitioning() {
    use crate::matrix::OptimMatrix;
    use rand::{thread_rng, Rng};
    for _ in 0..10 {
        let a: Vec<f32> = (0..thread_rng().gen_range(1000..1024))
            .map(|_| rand::random::<f32>())
            .collect();
        let b: Vec<f32> = (0..thread_rng().gen_range(a.len()..1024))
            .map(|_| rand::random::<f32>())
            .collect();
        let res = diamond_partitioning_gpu::<OptimMatrix>(
            &a,
            &b,
            f32::INFINITY,
            |a, b, i, j, x, y, z| {
                let dist = (a[i] - b[j]).abs();
                dist + z.min(x.min(y))
            },
        );
    }
}

pub fn diamond_partitioning_gpu<M: DiagonalMatrix>(
    a: &[f32],
    b: &[f32],
    init_val: f32,
    dist_lambda: impl Fn(&[f32], &[f32], usize, usize, f32, f32, f32) -> f32 + Copy,
) -> f32 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    let new_a_len = next_multiple_of_n(a.len(), DIAMOND_SIZE);
    let new_b_len = next_multiple_of_n(b.len(), DIAMOND_SIZE);

    let mut a_padded = vec![0.0; new_a_len];
    let mut b_padded = vec![0.0; new_b_len];

    a_padded[..a.len()].copy_from_slice(a);
    b_padded[..b.len()].copy_from_slice(b);

    diamond_partitioning_gpu_::<M>(a.len(), b.len(), init_val, |i, j, x, y, z| {
        dist_lambda(&a_padded, &b_padded, i, j, x, y, z)
    })
}

pub fn diamond_partitioning_gpu_<M: DiagonalMatrix>(
    a_len: usize,
    b_len: usize,
    init_val: f32,
    dist_lambda: impl Fn(usize, usize, f32, f32, f32) -> f32 + Copy,
) -> f32 {
    let padded_a_len = next_multiple_of_n(a_len, DIAMOND_SIZE);
    let padded_b_len = next_multiple_of_n(b_len, DIAMOND_SIZE);

    let mut matrix = M::new(padded_a_len, padded_b_len, init_val);

    matrix.set_diagonal_cell(0, 0, 0.0);

    let a_diamonds = padded_a_len.div_ceil(DIAMOND_SIZE);
    let b_diamonds = padded_b_len.div_ceil(DIAMOND_SIZE);
    let rows_count = (padded_a_len + padded_b_len).div_ceil(DIAMOND_SIZE) - 1;

    let mut diamonds_count = 1;
    let mut first_coord = -(DIAMOND_SIZE as isize);
    let mut a_start = 0;
    let mut b_start = 0;

    // Number of kernel calls
    for i in 0..rows_count {
        // Single kernel call
        for j in 0..diamonds_count {
            let diag_start = first_coord + ((j * DIAMOND_SIZE) as isize) * 2;
            let d_a_start = a_start - j * DIAMOND_SIZE;
            let d_b_start = b_start + j * DIAMOND_SIZE;

            let alen = a_len - d_a_start;
            let blen = b_len - d_b_start;

            // Single warp
            warp_kernel(
                &mut matrix,
                i * DIAMOND_SIZE,
                d_a_start,
                d_b_start,
                diag_start + (DIAMOND_SIZE as isize),
                (DIAMOND_SIZE * 2 + 1).min(alen + blen + 1),
                dist_lambda,
            );
        }

        if i < (a_diamonds - 1) {
            diamonds_count += 1;
            first_coord -= DIAMOND_SIZE as isize;
            a_start += DIAMOND_SIZE;
        } else if i < (b_diamonds - 1) {
            first_coord += DIAMOND_SIZE as isize;
            b_start += DIAMOND_SIZE;
        } else {
            diamonds_count -= 1;
            first_coord += DIAMOND_SIZE as isize;
            b_start += DIAMOND_SIZE;
        }
    }

    let (rx, cx) = M::index_mat_to_diag(a_len, b_len);

    matrix.get_diagonal_cell(rx, cx)
}

pub fn warp_kernel<M: DiagonalMatrix>(
    matrix: &mut M,
    d_offset: usize,
    a_start: usize,
    b_start: usize,
    diag_mid: isize,
    diag_count: usize,
    dist_lambda: impl Fn(usize, usize, f32, f32, f32) -> f32,
) {
    let mut i = a_start;
    let mut j = b_start;
    let mut s = diag_mid;
    let mut e = diag_mid;

    for d in 2..diag_count {
        for warp in 0..64 {
            let k = (warp * 2) as isize + s;
            if k <= e {
                let i1 = i - warp;
                let j1 = j + warp;

                let dleft = matrix.get_diagonal_cell(d_offset + d - 1, k - 1);
                let ddiag = matrix.get_diagonal_cell(d_offset + d - 2, k);
                let dup = matrix.get_diagonal_cell(d_offset + d - 1, k + 1);

                matrix.set_diagonal_cell(d_offset + d, k, dist_lambda(i1, j1, dleft, ddiag, dup));
            }
        }
        // Warp synchronize

        if d <= DIAMOND_SIZE {
            i += 1;
            s -= 1;
            e += 1;
        } else {
            j += 1;
            s += 1;
            e -= 1;
        }
    }
}
