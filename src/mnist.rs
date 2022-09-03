use mnist::Mnist;
use mnist::MnistBuilder;

pub struct MnistImage {
    pub image: ndarray::Array1<f32>,
    pub label: ndarray::Array1<f32>,
}

pub fn get_mnist() -> (Vec<MnistImage>, Vec<MnistImage>, Vec<MnistImage>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_one_hot()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let trn = trn_img
        .chunks(28 * 28)
        .zip(trn_lbl.chunks(10))
        .map(|(image, label)| {
            let image = image.iter().map(|&x| (x as f32) / 255.0).collect();
            let label = label.iter().map(|&x| x as f32).collect();

            MnistImage {
                image: ndarray::Array1::from_shape_vec(28 * 28, image).unwrap(),
                label: ndarray::Array1::from_shape_vec(10, label).unwrap(),
            }
        })
        .collect();

    let val = tst_img
        .chunks(28 * 28)
        .zip(tst_lbl.chunks(10))
        .map(|(image, label)| {
            let image = image.iter().map(|&x| (x as f32) / 255.0).collect();
            let label = label.iter().map(|&x| x as f32).collect();

            MnistImage {
                image: ndarray::Array1::from_shape_vec(28 * 28, image).unwrap(),
                label: ndarray::Array1::from_shape_vec(10, label).unwrap(),
            }
        })
        .collect();

    let tst = tst_img
        .chunks(28 * 28)
        .zip(tst_lbl.chunks(10))
        .map(|(image, label)| {
            let image = image.iter().map(|&x| (x as f32) / 255.0).collect();
            let label = label.iter().map(|&x| x as f32).collect();

            MnistImage {
                image: ndarray::Array1::from_shape_vec(28 * 28, image).unwrap(),
                label: ndarray::Array1::from_shape_vec(10, label).unwrap(),
            }
        })
        .collect();

    (trn, val, tst)
}
