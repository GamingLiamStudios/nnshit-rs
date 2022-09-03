#![feature(int_roundings)]
mod network;

extern crate blas_src;

mod mnist;

use network::misc::*;
use network::*;

use rand::prelude::*;

fn main() {
    println!("Hello, world!");

    let (mut trn, mut val, mut tst) = mnist::get_mnist();

    let mut rng = thread_rng();
    trn.shuffle(&mut rng);
    val.shuffle(&mut rng);
    tst.shuffle(&mut rng);

    println!("Training set: {} images", trn.len());
    println!("Validation set: {} images", val.len());
    println!("Test set: {} images", tst.len());

    let network = NetworkBuilder::new(28 * 28)
        .add_layer(300, activation::SIGMOID)
        .add_layer(30, activation::SIGMOID)
        .finalize(10, activation::SIGMOID, cost::MSE);

    println!("Running Test set");
    let mut cost = 0.0;
    for (i, mimg) in tst.chunks(500).enumerate() {
        let (x, y): (Vec<_>, Vec<_>) = mimg
            .iter()
            .map(|x| (x.image.clone(), x.label.clone()))
            .unzip();
        let batch = network.test(&x, &y);

        cost += batch;
        println!("Batch {} / {}: {}", i * 500, tst.len(), batch);
    }
    cost /= tst.len().div_ceil(500) as f32;

    println!("Test set cost: {}", cost);

    let input = trn[0].image.clone();
    let output = network.forward(&input, classifier::ARGMAX);

    println!("{:?}", output);
    println!("{:?}", trn[0].label);
}
