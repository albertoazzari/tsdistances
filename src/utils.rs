pub fn derivate(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut x_d = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        x_d.push(vec![0.0; x[i].len()]);
    }
    for i in 0..x.len() {
        for j in 1..x[i].len() - 1 {
            x_d[i][j] = ((x[i][j] - x[i][j - 1]) + (x[i][j + 1] - x[i][j - 1]) / 2.0) / 2.0;
        }
        x_d[i][0] = x_d[i][1];
        x_d[i][x[i].len() - 1] = x_d[i][x[i].len() - 2];
    }
    x_d
}

const WEIGHT_MAX: f64 = 1.0;
pub fn dtw_weights(len: usize, g: f64) -> Vec<f64> {
    let mut weights = vec![0.0; len];
    let half_len = len as f64 / 2.0;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + std::f64::consts::E.powf(-g * (i as f64 - half_len)));
    }
    weights
}

const MSM_C: f64 = 1.0;
pub fn msm_cost_function(x_i: f64, x_i_1: f64, y_j: f64) -> f64 {
    if (x_i >= x_i_1 && x_i <= y_j) || (x_i_1 >= x_i && x_i >= y_j) {
        MSM_C
    } else {
        MSM_C + (x_i - x_i_1).abs().min((x_i - y_j).abs())
    }
}