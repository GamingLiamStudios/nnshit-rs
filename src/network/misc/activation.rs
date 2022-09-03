#[derive(Clone, Copy)]
pub struct Activation {
    pub compute: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
}

pub const NONE: Activation = Activation {
    compute: |x: f32| -> f32 { x },
    derivative: |_: f32| -> f32 { 1.0 },
};

pub const SIGMOID: Activation = Activation {
    compute: |x: f32| -> f32 { 1.0 / (1.0 + (-x).exp()) },
    derivative: |x: f32| -> f32 {
        let y = 1.0 / (1.0 + (-x).exp());
        y * (1.0 - y)
    },
};

pub const RELU: Activation = Activation {
    compute: |x: f32| -> f32 { x.max(0.0) },
    derivative: |x: f32| -> f32 { (x > 0.0) as i32 as _ },
};

pub const TANH: Activation = Activation {
    compute: |x: f32| -> f32 { x.tanh() },
    derivative: |x: f32| -> f32 { 1.0 - x.tanh().powi(2) },
};
