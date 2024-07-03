use crate::diagonal;

const DIAMOND_SIZE: usize = 64;

#[test]
fn test_diamond_partitioning() {
    let a: Vec<f64> = (0..31).map(|_| rand::random::<f64>()).collect();
    let b: Vec<f64> = (0..63).map(|_| rand::random::<f64>()).collect();

    // let start = std::time::Instant::now();
    let res = diamond_partitioning(
        a.as_slice(),
        b.as_slice(),
        f64::INFINITY,
        |i, j, x, y, z| {
            let dist = (a.get(i).copied().unwrap_or(0.0) - b.get(j).copied().unwrap_or(0.0)).powi(2);
            dist + x.min(y.min(z))
        },
    );
    // println!("Time: {:?}", start.elapsed());
    println!("\n");
    // let start = std::time::Instant::now();
    let r2 = diagonal::diagonal_distance(&a, &b, f64::INFINITY, |i, j, x, y, z| {
        let dist = (a[i] - b[j]).powi(2);
        dist + z.min(x.min(y))
    });
    // println!("Time: {:?}", start.elapsed());
    assert_eq!(res, r2);
}

pub fn diamond_partitioning(
    a: &[f64],
    b: &[f64],
    init_val: f64,
    dist_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64 + Copy,
) -> f64 {
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };

    // let (a, a_mask) = {
    //     let next_power_of_two = a.len().next_power_of_two();
    //     let mask = next_power_of_two - 1;
    //     // pad a
    //     let a = a
    //         .iter()
    //         .copied()
    //         .chain(std::iter::repeat(0.0))
    //         .take(next_power_of_two)
    //         .collect::<Vec<f64>>();
    //     (a, mask)
    // };

    // let (b, b_mask) = {
    //     let next_power_of_two = b.len().next_power_of_two();
    //     let mask = next_power_of_two - 1;
    //     // pad b
    //     let b = b
    //         .iter()
    //         .copied()
    //         .chain(std::iter::repeat(0.0))
    //         .take(next_power_of_two)
    //         .collect::<Vec<f64>>();
    //     (b, mask)
    // };

    let next_power_of_two = (2 * a.len().next_power_of_two());//.min(DIAMOND_SIZE * 2);
    let mask = next_power_of_two - 1;

    let mut diagonal = vec![init_val; next_power_of_two];

    let offset: usize = (a.len().div_ceil(DIAMOND_SIZE) + 2) * DIAMOND_SIZE;

    diagonal[offset & mask] = 0.0;

    let a_diamonds = a.len() / DIAMOND_SIZE;
    let b_diamonds = b.len() / DIAMOND_SIZE;
    let rows_count = a_diamonds + b_diamonds + 1;

    let mut diamonds_count = 1;
    let mut first_coord = offset - DIAMOND_SIZE;
    let mut a_start = 0;
    let mut b_start = 0;
    let mut s = 0;

    for i in 0..rows_count {
        // for d in 1..DIAMOND_SIZE {
        //     diagonal[(i * DIAMOND_SIZE + offset + d) & mask] = init_val;
        // }
        // if a_start==0 {
        //     diagonal[(diag_mid + d) & mask] = f64::INFINITY;
        // } else if b_start==0 {
        //     diagonal[(diag_mid - d) & mask] = f64::INFINITY;
        // }
        // println!("Row: {} {}", i, diamonds_count);
        for j in 0..diamonds_count {
            let diag_start = first_coord + j * DIAMOND_SIZE * 2;
            let d_a_start = a_start - j * DIAMOND_SIZE;
            let d_b_start = b_start + j * DIAMOND_SIZE;
            s = diagonal_distance_v2(
                &mut diagonal,
                d_a_start,
                a.len(),
                d_b_start,
                b.len(),
                diag_start + DIAMOND_SIZE,
                mask,
                dist_lambda,
            );

            // println!(
            //     "Diamond: {} {} {}",
            //     diag_start as isize - offset as isize + DIAMOND_SIZE as isize,
            //     d_a_start,
            //     d_b_start
            // );
        }

        if i < a_diamonds {
            diamonds_count += 1;
            first_coord -= DIAMOND_SIZE;
            a_start += DIAMOND_SIZE;
        } else if i < b_diamonds {
            first_coord += DIAMOND_SIZE;
            b_start += DIAMOND_SIZE;
        } else {
            diamonds_count -= 1;
            first_coord += DIAMOND_SIZE;
            b_start += DIAMOND_SIZE;
        }
    }
    // diagonal[offset & mask]
    diagonal[s]
}

pub fn diagonal_distance_v2(
    diagonal: &mut [f64],
    a_start: usize,
    a_len: usize,
    b_start: usize,
    b_len: usize,
    diag_mid: usize,
    mask: usize,
    dist_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64,
) -> usize {
    let mut i = a_start;
    let mut j = b_start;
    let mut s = diag_mid;
    let mut e = diag_mid;

    for d in 2..(DIAMOND_SIZE * 2 + 1).min(a_len + b_len + 1 - (a_start + b_start)) {
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

        if d <= DIAMOND_SIZE.min(a_len - a_start) {
            i += 1;
            s -= 1;
            e += 1;
        } else if d <= DIAMOND_SIZE.min(b_len - b_start) {
            j += 1;
            s += 1;
            e += 1;
        } else{
            j += 1;
            s += 1;
            e -= 1;
        }
    }
    (s-1)&mask
}
