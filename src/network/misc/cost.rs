pub struct Cost {
    pub compute: fn(&ndarray::Array1<f32>, &ndarray::Array1<f32>) -> f32,
    pub derivative: fn(&ndarray::Array1<f32>, &ndarray::Array1<f32>) -> ndarray::Array1<f32>,
}

// Mean Squared Error
pub const MSE: Cost = Cost {
    compute: |y: &ndarray::Array1<f32>, t: &ndarray::Array1<f32>| -> f32 {
        let mut sum = 0.0;
        for (y_i, t_i) in y.iter().zip(t.iter()) {
            sum += (y_i - t_i).powi(2);
        }
        sum / (2.0 * y.len() as f32)
    },
    derivative: |y: &ndarray::Array1<f32>, t: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
        (y - t) / y.len() as f32
    },
};
