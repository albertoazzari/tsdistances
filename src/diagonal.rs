use crate::distances;

// previous[j] = z
// current[j-1] = x
// previous[j-1] = y

fn diagonal_distance(
    a: &[f64],
    b: &[f64],
    init_val: f64,
    dist_lambda: impl Fn(f64, f64, f64, f64, f64) -> f64,
) -> f64 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
    // let a = b;

    let square_len = a.len();

    let mut matrix = [
        vec![init_val; square_len],
        vec![init_val; square_len],
        vec![init_val; square_len],
    ];
    matrix[0][0] = 0.0;

    // println!("{:?}", matrix[0]);
    // println!("{:?}", matrix[1]);

    for i in 2..(a.len() + b.len() - 1) {
        matrix[i % 3][0] = init_val;

        let a_start = i.min(a.len() - 1);
        let b_start = i.saturating_sub(a.len() - 1);

        let j_start = if i >= a.len() { 0 } else { 1 };

        let xz_offset = if i >= a.len() { 1 } else { 0 };
        let y_offset = if i >= a.len() { 1 } else { 0 } + if i > a.len() { 1 } else { 0 };

        let max_len = a.len().min(i).min(a.len() + b.len() - i - 1);

        for j in j_start..max_len {
            let x = matrix[(i - 1) % 3][j + xz_offset - 1];
            let y = matrix[(i - 2) % 3][j + y_offset - 1];
            let z = matrix[(i - 1) % 3][j + xz_offset];

            matrix[i % 3][j] = dist_lambda(a[a_start - j], b[b_start + j], x, y, z);
        }

        // println!("{:?}", matrix[i % 3])
    }

    matrix[(a.len() + b.len() - 2) % 3][0]
}

#[pyo3::pyfunction]
#[pyo3(signature = (x1, x2=None, n_jobs=-1))]
pub fn dtw_diag(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    n_jobs: i32,
) -> pyo3::PyResult<Vec<Vec<f64>>> {
    let distance_matrix = distances::compute_distance(
        |a, b| {
            diagonal_distance(a, b, f64::INFINITY, |a, b, x, y, z| {
                let dist = (a - b).powi(2);
                dist + z.min(x.min(y))
            })
        },
        x1,
        x2,
        n_jobs,
    );

    Ok(distance_matrix)
}

#[pyo3::pyfunction]
#[pyo3(signature = (x1, x2=None, gap_penalty = 1.0, n_jobs=-1))]
pub fn erp_diag(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    gap_penalty: f64,
    n_jobs: i32,
) -> pyo3::PyResult<Vec<Vec<f64>>> {
    let distance_matrix = distances::compute_distance(
        |a, b| {
            diagonal_distance(a, b, f64::INFINITY, |a, b, x, y, z| {
                (y + (a - b).abs())
                    .min((z + (a - gap_penalty).abs()).min(x + (b - gap_penalty).abs()))
            })
        },
        x1,
        x2,
        n_jobs,
    );

    Ok(distance_matrix)
}

#[pyo3::pyfunction]
#[pyo3(signature = (x1, x2=None, epsilon = 1.0, n_jobs=-1))]
pub fn lcss_diag(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    epsilon: f64,
    n_jobs: i32,
) -> pyo3::PyResult<Vec<Vec<f64>>> {
    let distance_matrix = distances::compute_distance(
        |a, b| {
            diagonal_distance(a, b, 0.0, |a, b, x, y, z| {
                let dist = (a - b).abs();
                if dist <= epsilon {
                    y + 1.0
                } else {
                    x.max(z)
                }
            })
        },
        x1,
        x2,
        n_jobs,
    );

    Ok(distance_matrix)
}

#[test]
fn test_lcss() {
    // Generate 100 random float vector
    let a: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();
    let b: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();

    let result_diag = lcss_diag(vec![a.to_vec()], Some(vec![b.to_vec()]), 1.0, 1).unwrap();
    let result = distances::lcss(vec![a.to_vec()], Some(vec![b.to_vec()]), 1.0, 1.0, false, 1).unwrap();
    assert_eq!(result_diag[0][0], result[0][0])
}

#[test]
fn test_erp() {
    // Generate 100 random float vector
    let a: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();
    let b: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();

    let result_diag = erp_diag(vec![a.to_vec()], Some(vec![b.to_vec()]), 1.0, 1).unwrap();
    let result = distances::erp(vec![a.to_vec()], Some(vec![b.to_vec()]), 1.0, 1.0, false, 1).unwrap();
    assert_eq!(result_diag[0][0], result[0][0])
}
