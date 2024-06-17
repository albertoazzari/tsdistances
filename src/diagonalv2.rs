use crate::distances;

fn diagonal_distance_v2(
    a: &[f64],
    b: &[f64],
    dist_lambda: impl Fn(f64, f64, f64, f64, f64) -> f64,
) -> f64 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    let mut diagonal = vec![f64::INFINITY; 1024];
    let mask = 0b1111111111;

    let mut i = 0;
    let mut j = 0;
    let mut s = a.len();
    let mut e = a.len();
    diagonal[a.len()] = 0.0;

    for d in 2..=(a.len() + b.len()) {
        diagonal[(a.len() + d) & mask] = f64::INFINITY;

        let mut i1 = i;
        let mut j1 = j;

        for k in (s..=e).step_by(2) {
            let x = diagonal[(k - 1) & mask];
            let y = diagonal[k & mask];
            let z = diagonal[(k + 1) & mask];

            diagonal[k & mask] = dist_lambda(a[i1], b[j1], x, y, z);
            i1 = i1.wrapping_sub(1);
            j1 += 1;
        }

        if d <= a.len() {
            i += 1;
            s -= 1;
            e += 1;
        } else if d <= b.len() {
            j += 1;
            s += 1;
            e += 1;
        } else {
            j += 1;
            s += 1;
            e -= 1;
        }
    }
    diagonal[s - 1]
}

#[test]
fn test_matrix() {
    // let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
    // let b: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let a: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();
    let b: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();

    let result = diagonal_distance_v2(&a, &b, |a, b, x, y, z| {
        let dist = (a - b).powi(2);
        dist + z.min(x.min(y))
    });

    let result2 = distances::dtw(vec![a.to_vec()], Some(vec![b.to_vec()]), 1.0, false, 1).unwrap();

    // // Generate 100 random float vector
    // // let b: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();

    // let result_diag = erp_diag(vec![a.to_vec()], Some(vec![b.to_vec()]), 1.0, 1).unwrap();
    // let result =
    //     distances::erp(vec![a.to_vec()], Some(vec![b.to_vec()]), 1.0, 1.0, false, 1).unwrap();
    // assert_eq!(result_diag[0][0], result[0][0])
}

#[test]
fn test_erp() {
    // let a: Vec<f64> = (0..100).map(|i| i as f64 / 123321.021).collect();
    // let b: Vec<f64> = ((0..100).rev()).map(|i| i as f64 / 123321.021).collect();

    // Generate 100 random float vector
    let a: Vec<f64> = (0..10).map(|_| rand::random::<f64>()).collect();
    let b: Vec<f64> = (0..10).map(|_| rand::random::<f64>()).collect();

    let gap_penalty = 0.0;

    let result_diag = diagonal_distance_v2(&a, &b, |a, b, x, y, z| {
        (y + (a - b).abs()).min((z + (a - gap_penalty).abs()).min(x + (b - gap_penalty).abs()))
    });
    let result = distances::erp(
        vec![a.to_vec()],
        Some(vec![b.to_vec()]),
        gap_penalty,
        1.0,
        false,
        1,
    )
    .unwrap();
    println!("{:?} {:?}", result_diag, result[0][0]);
    assert_eq!(result_diag, result[0][0])
}
