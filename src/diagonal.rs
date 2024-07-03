use std::mem;

pub fn diagonal_distance(
    a: &[f64],
    b: &[f64],
    init_val: f64,
    dist_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64,
) -> f64 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    println!("a.len {}, b.len {}", a.len(), b.len());
    // assert_eq!(a.as_ptr(), c.as_ptr());
    // assert_eq!(a.as_ptr(), d.as_ptr());

    let next_power_of_two = 2 * a.len().next_power_of_two();

    let offset = a.len();

    let mut diagonal = vec![init_val; next_power_of_two];
    let mask = next_power_of_two - 1;

    let mut i = 0;
    let mut j = 0;
    let mut s = offset;
    let mut e = offset;
    diagonal[offset & mask] = 0.0;
    for d in 2..(a.len() + b.len() + 1) {
        diagonal[(offset + d) & mask] = init_val;

        let mut i1 = i;
        let mut j1 = j;

        for k in (s..e + 1).step_by(2) {
            let x = diagonal[(k - 1) & mask];
            let y = diagonal[k & mask];
            let z = diagonal[(k + 1) & mask];

            diagonal[k & mask] = dist_lambda(i1, j1, x, y, z);
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

    // println!("{:?}", diagonal);
    diagonal[(s - 1) & mask]
}

#[test]
fn test_matrix() {
    let a: Vec<f64> = (0..1000).map(|_| rand::random::<f64>()).collect();
    let b: Vec<f64> = (0..1000).map(|_| rand::random::<f64>()).collect();

    let result = diagonal_distance(
        a.as_slice(),
        b.as_slice(),
        f64::INFINITY,
        |i, j, x, y, z| (a[i] - b[j]).powi(2) + x.min(y.min(z)),
    );

    assert!(result >= 0.0);
}
