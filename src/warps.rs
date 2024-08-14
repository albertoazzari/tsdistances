use crate::{matrix::Matrix, utils::next_multiple_of_n};
const DIAMOND_SIZE: usize = 64;

#[test]
fn test_diamond_partitioning() {
    use crate::diagonal::diagonal_distance;
    use crate::matrix::DiagonalMatrix;
    use tsdistances_gpu::device::get_best_gpu;
    use tsdistances_gpu::SingleBatchMode;

    let rep = 10;
    let epsilon = 0.1;
    let (mut time1, mut time2, mut time3) = (0, 0, 0);
    let (mut r1, mut r2, mut r3) = (0.0, 0.0, 0.0);
    let device = get_best_gpu();
    println!("Device: {:#?}", device.info().unwrap());

    for _ in 0..rep {
        let a: Vec<f64> = (0..20000).map(|_| rand::random::<f64>()).collect();
        let b: Vec<f64> = (0..20000).map(|_| rand::random::<f64>()).collect();

        let start1 = std::time::Instant::now();
        r1 += diamond_partitioning::<DiagonalMatrix>(&a, &b, 0.0, |a, b, i, j, x, y, z| {
            let dist = (a[i] - b[j]).abs();
            if dist <= epsilon {
                y + 1.0
            } else {
                x.max(z)
            }
        });
        let end1 = start1.elapsed();
        time1 += end1.as_micros();

        let start2 = std::time::Instant::now();
        r2 += diagonal_distance::<DiagonalMatrix>(&a, &b, 0.0, 1.0, |a, b, i, j, x, y, z| {
            let dist = (a[i] - b[j]).abs();
            if dist <= epsilon {
                y + 1.0
            } else {
                x.max(z)
            }
        });
        let end2 = start2.elapsed();
        time2 += end2.as_micros();

        let start3 = std::time::Instant::now();
        r3 += tsdistances_gpu::lcss::<SingleBatchMode>(device.clone(), &a, &b, epsilon);
        let end3 = start3.elapsed();
        time3 += end3.as_micros();
    }

    println!(
        "TIME:\n\tDiamond Partitioning: {:.4} ms, \n\tDiagonal Distance: {:.4} ms, \n\tGPU: {:.4} ms",
        time1 as f64 / rep as f64 / 1000.0,
        time2 as f64 / rep as f64 / 1000.0,
        time3 as f64 / rep as f64 / 1000.0
    );
    println!(
        "RES:\n\tDiamond Partitioning: {:.4}, \n\tDiagonal Distance: {:.4}, \n\tGPU: {:.4}",
        r1 / rep as f64,
        r2 / rep as f64,
        r3 / rep as f64
    );
}

pub fn diamond_partitioning<M: Matrix>(
    a: &[f64],
    b: &[f64],
    init_val: f64,
    dist_lambda: impl Fn(&[f64], &[f64], usize, usize, f64, f64, f64) -> f64 + Copy,
) -> f64 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    let new_a_len = next_multiple_of_n(a.len(), DIAMOND_SIZE);
    let new_b_len = next_multiple_of_n(b.len(), DIAMOND_SIZE);

    let mut a_padded = vec![0.0; new_a_len];
    let mut b_padded = vec![0.0; new_b_len];

    a_padded[..a.len()].copy_from_slice(a);
    b_padded[..b.len()].copy_from_slice(b);

    diamond_partitioning_::<M>(a.len(), b.len(), init_val, |i, j, x, y, z| {
        dist_lambda(&a_padded, &b_padded, i, j, x, y, z)
    })
}

pub fn diamond_partitioning_<M: Matrix>(
    a_len: usize,
    b_len: usize,
    init_val: f64,
    dist_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64 + Copy,
) -> f64 {
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

    for i in 0..rows_count {
        for j in 0..diamonds_count {
            let diag_start = first_coord + ((j * DIAMOND_SIZE) as isize) * 2;
            let d_a_start = a_start - j * DIAMOND_SIZE;
            let d_b_start = b_start + j * DIAMOND_SIZE;

            let alen = a_len - d_a_start;
            let blen = b_len - d_b_start;

            diagonal_distance_v2(
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

pub fn diagonal_distance_v2<M: Matrix>(
    matrix: &mut M,
    d_offset: usize,
    a_start: usize,
    b_start: usize,
    diag_mid: isize,
    diag_count: usize,
    dist_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64,
) {
    let mut i = a_start;
    let mut j = b_start;
    let mut s = diag_mid;
    let mut e = diag_mid;

    for d in 2..diag_count {
        let mut i1 = i;
        let mut j1 = j;

        for k in (s..e + 1).step_by(2) {
            let dleft = matrix.get_diagonal_cell(d_offset + d - 1, k - 1);
            let ddiag = matrix.get_diagonal_cell(d_offset + d - 2, k);
            let dup = matrix.get_diagonal_cell(d_offset + d - 1, k + 1);

            matrix.set_diagonal_cell(d_offset + d, k, dist_lambda(i1, j1, dleft, ddiag, dup));
            i1 = i1.wrapping_sub(1);
            j1 += 1;
        }

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
