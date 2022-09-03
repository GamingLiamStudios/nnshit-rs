use crate::misc::activation::Activation;
use rand::prelude::*;

pub struct Layer {
    weights: ndarray::Array2<f32>,
    biases: ndarray::Array2<f32>,

    activation: Activation,
}

impl Layer {
    pub fn new(shape: (usize, usize), activation: Activation) -> Self {
        let mut rng = thread_rng();

        let (input, output) = shape;

        let weights = ndarray::Array2::from_shape_fn((output, input), |_| {
            rng.sample::<f32, _>(rand_distr::StandardNormal)
        });
        let biases = ndarray::Array2::zeros((output, 1));

        Self {
            weights,
            biases,
            activation,
        }
    }

    #[inline]
    pub fn forward(&self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        let mut a = self.weights.dot(input) + &self.biases;
        a.mapv_inplace(self.activation.compute);
        a
    }
}
