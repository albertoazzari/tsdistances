use crate::matrix::Matrix;

pub fn diagonal_distance<M: Matrix>(
    a: &[f64],
    b: &[f64],
    init_val: f64,
    dist_lambda: impl Fn(&[f64], &[f64], usize, usize, f64, f64, f64) -> f64 + Copy,
) -> f64 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    diagonal_distance_::<M>(a.len(), b.len(), init_val, |i, j, x, y, z| {
        dist_lambda(&a, &b, i, j, x, y, z)
    })
}

fn diagonal_distance_<M: Matrix>(
    a_len: usize,
    b_len: usize,
    init_val: f64,
    dist_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64,
) -> f64 {
    let mut matrix = M::new(a_len, b_len, init_val);

    let mut i = 0;
    let mut j = 0;
    let mut s = 0;
    let mut e = 0;
    matrix.set_diagonal_cell(0, 0, 0.0);

    for d in 2..(a_len + b_len + 1) {
        matrix.set_diagonal_cell(d, d as isize, init_val);

        let mut i1 = i;
        let mut j1 = j;

        for k in (s..e + 1).step_by(2) {
            let dleft = matrix.get_diagonal_cell(d - 1, k - 1);
            let ddiag = matrix.get_diagonal_cell(d - 2, k);
            let dup = matrix.get_diagonal_cell(d - 1, k + 1);

            matrix.set_diagonal_cell(d, k, dist_lambda(i1, j1, dleft, ddiag, dup));
            i1 = i1.wrapping_sub(1);
            j1 += 1;
        }

        if d <= a_len {
            i += 1;
            s -= 1;
            e += 1;
        } else if d <= b_len {
            j += 1;
            s += 1;
            e += 1;
        } else {
            j += 1;
            s += 1;
            e -= 1;
        }
    }
    let (rx, cx) = M::index_mat_to_diag(a_len, b_len);
    matrix.get_diagonal_cell(rx, cx)
}