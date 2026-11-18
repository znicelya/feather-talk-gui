use burn::tensor::{Int, Tensor, TensorData};
/// Separable align-corners interpolation; select's additive backward stays on the execution device.
pub fn resize_bilinear_align_corners(mut input: Tensor<4>, output_hw: [usize; 2]) -> Tensor<4> {
    assert!(output_hw.iter().all(|&n| n > 0));
    for (axis, size) in [(2, output_hw[0]), (3, output_hw[1])] {
        let source = input.dims()[axis];
        let scale = if size > 1 {
            (source - 1) as f32 / (size - 1) as f32
        } else {
            0.0
        };
        let positions: Vec<f32> = (0..size).map(|i| i as f32 * scale).collect();
        let lo: Vec<i32> = positions.iter().map(|x| x.floor() as i32).collect();
        let hi: Vec<i32> = lo.iter().map(|&i| (i + 1).min(source as i32 - 1)).collect();
        let weights: Vec<f32> = positions
            .iter()
            .zip(&lo)
            .map(|(x, l)| x - *l as f32)
            .collect();
        let device = input.device();
        let lower = Tensor::<1, Int>::from_data(TensorData::new(lo, [size]), &device);
        let upper = Tensor::<1, Int>::from_data(TensorData::new(hi, [size]), &device);
        let mut shape = [1; 4];
        shape[axis] = size;
        let weight = Tensor::<4>::from_data(TensorData::new(weights, shape), &device);
        input = input.clone().select(axis, lower) * (weight.clone().neg() + 1.0)
            + input.select(axis, upper) * weight;
    }
    input
}
