mod layer;
pub mod misc;

pub use self::layer::*;
use rand::seq::SliceRandom;

use crate::mnist::MnistImage;

use self::misc::activation::Activation;
use self::misc::cost::Cost;
use self::misc::*;

pub struct Network {
    layers: Vec<Layer>,
    cost: Cost,
}

pub struct NetworkBuilder {
    input: usize,
    layers: Vec<(usize, Activation)>,
}

impl NetworkBuilder {
    pub fn new(input: usize) -> Self {
        Self {
            input,
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, size: usize, activation: Activation) -> &mut Self {
        self.layers.push((size, activation));
        self
    }

    pub fn finalize(&self, output: usize, activation: Activation, cost: Cost) -> Network {
        if self.layers.is_empty() {
            panic!("No layers added to network");
        }

        let mut layers = Vec::new();
        let mut prev_size = self.input;
        for (size, activation) in self.layers.iter() {
            layers.push(Layer::new((prev_size, *size), *activation));
            prev_size = *size;
        }
        layers.push(Layer::new((prev_size, output), activation));
        Network { layers, cost }
    }
}

impl Network {
    /// Takes in an `input` and an optional `classifier` and returns the output of the network.
    pub fn forward(
        &self,
        input: &ndarray::Array1<f32>,
        classifier: fn(&ndarray::Array1<f32>) -> ndarray::Array1<f32>,
    ) -> ndarray::Array1<f32> {
        let mut b = input.clone();
        {
            let mut a = ndarray::Array2::from_shape_vec((b.len(), 1), b.to_vec()).unwrap();

            for layer in &self.layers {
                a = layer.forward(&a);
            }

            b = ndarray::Array1::from_iter(a.iter().cloned());
        }

        classifier(&b)
    }

    /// Takes in a `test_set` as a vector of `input`s and `expected` results, and returns the
    /// average `cost` of the network on the `test_set`.
    pub fn test(
        &self,
        input_set: &Vec<ndarray::Array1<f32>>,
        expected: &Vec<ndarray::Array1<f32>>,
    ) -> f32 {
        let mut cost = 0.0;
        for (i, input) in input_set.iter().enumerate() {
            let output = self.forward(input, classifier::NONE);
            cost += (self.cost.compute)(&output, &expected[i]);
        }
        cost / input_set.len() as f32
    }

    // Calculate the derivatives of the
    pub fn back(&self, input: ndarray::Array1<f32>, expected: ndarray::Array1<f32>) {}

    // Calculate how much each weight & bias affects the cost function (derivatives)
    // Multiply that by `learn_rate`
    // Add? it to the current network & do it all again
    // https://youtu.be/hfMk-kjRv4c?t=2123
    pub fn train(&self, mut train_data: Vec<MnistImage>, batch_size: usize, learn_rate: f32) {
        let mut rng = rand::thread_rng();

        train_data.shuffle(&mut rng);
        for batch in train_data.chunks(batch_size).into_iter() {}
    }
}
