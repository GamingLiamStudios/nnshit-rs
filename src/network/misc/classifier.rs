// The const shit to keep consistency with the other modules

type Classifier = fn(&ndarray::Array1<f32>) -> ndarray::Array1<f32>;

pub const NONE: Classifier = |x: &ndarray::Array1<f32>| -> ndarray::Array1<f32> { x.clone() };

pub const ARGMAX: Classifier = |x: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
    let mut y = ndarray::Array1::zeros(x.len());
    let mut max = 0.0;
    let mut max_index = 0;
    for (i, &x_i) in x.iter().enumerate() {
        if x_i > max {
            max = x_i;
            max_index = i;
        }
    }
    y[max_index] = 1.0;
    y
};

pub const SOFTMAX: Classifier = |x: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
    let mut y = ndarray::Array1::zeros(x.len());
    let mut sum = 0.0;
    for (i, &x_i) in x.iter().enumerate() {
        y[i] = x_i.exp();
        sum += y[i];
    }
    y / sum
};
