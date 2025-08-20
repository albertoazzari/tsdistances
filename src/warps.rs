use crate::{matrix::Matrix, utils::next_multiple_of_n};

const DIAMOND_SIZE: usize = 64;

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

            diagonal_distance(
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

pub fn diagonal_distance<M: Matrix>(
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

            let value = dist_lambda(i1, j1, dleft, ddiag, dup);
            matrix.set_diagonal_cell(d_offset + d, k, value);

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
