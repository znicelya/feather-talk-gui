use crate::{
    device::ExecutionDevice,
    ops::{
        attention::SpatialAudioAttention,
        batch_norm::{BatchNorm, group_norm},
        resize::resize_bilinear_align_corners,
    },
    reference::{Reference, error},
};
use burn::tensor::{
    Tensor,
    module::{avg_pool2d, conv1d, conv2d},
    ops::{AvgPoolOptions, ConvOptions},
};
use feathertalk_core::Result;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path};
#[derive(Debug, Serialize)]
pub struct ParityCheck {
    pub name: String,
    pub passed: bool,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
}
#[derive(Debug, Serialize)]
pub struct OperatorReport {
    pub device: feathertalk_core::DeviceDescriptor,
    pub checks: Vec<ParityCheck>,
}
fn check<const D: usize>(
    checks: &mut Vec<ParityCheck>,
    case: &str,
    r: &Reference,
    key: &str,
    value: Tensor<D>,
) -> Result<()> {
    let data = value
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| error(format!("{e:?}")))?;
    let expected = &r.values[key].1;
    let tolerance = r
        .case
        .tensor_tolerances
        .get(key)
        .unwrap_or(&r.case.tolerance);
    let mut passed = data.len() == expected.len();
    let mut abs = 0f64;
    let mut rel = 0f64;
    for (&actual, &expected) in data.iter().zip(expected) {
        let diff = (actual as f64 - expected as f64).abs();
        passed &=
            actual.is_finite() && diff <= tolerance.atol + tolerance.rtol * (expected as f64).abs();
        abs = abs.max(diff);
        rel = rel.max(diff / (expected as f64).abs().max(1e-30));
    }
    if key.contains("gradient") && expected.iter().any(|v| *v != 0.0) {
        passed &= data.iter().any(|v| *v != 0.0);
    }
    checks.push(ParityCheck {
        name: format!("{case}/{key}"),
        passed,
        max_abs_error: abs,
        max_rel_error: rel,
    });
    Ok(())
}
pub fn run_operator_checks(device: &ExecutionDevice, reference: &Path) -> Result<OperatorReport> {
    let mut checks = vec![];
    let d = device.device.clone().autodiff();
    let name = "resize_align_corners";
    let r = Reference::load(reference, name)?;
    let x = r.tensor::<4>("input.0", &d).require_grad();
    let size = &r.case.configuration["size"];
    let y = resize_bilinear_align_corners(
        x.clone(),
        [
            size[0].as_u64().unwrap() as usize,
            size[1].as_u64().unwrap() as usize,
        ],
    );
    let objective = (y.clone() * r.tensor("upstream_gradient", &d)).sum();
    check(&mut checks, name, &r, "output", y)?;
    check(&mut checks, name, &r, "objective", objective.clone())?;
    let grads = objective.backward();
    check(
        &mut checks,
        name,
        &r,
        "input_gradient.0",
        x.grad(&grads).unwrap(),
    )?;
    for name in ["batch_norm_nonlinear", "group_norm_nonlinear"] {
        let r = Reference::load(reference, name)?;
        let x = r.tensor::<4>("input.0", &d).require_grad();
        let [b, c, h, w] = x.dims();
        let weight = r.tensor::<1>("state.weight", &d).require_grad();
        let bias = r.tensor::<1>("state.bias", &d).require_grad();
        let y = if name.starts_with("batch") {
            let mut model = BatchNorm {
                weight: weight.clone(),
                bias: bias.clone(),
                running_mean: Tensor::zeros([c], &d),
                running_var: Tensor::ones([c], &d),
                num_batches_tracked: 0,
                momentum: 0.1,
                epsilon: 1e-5,
            };
            let y = model.forward(x.clone(), true);
            check(
                &mut checks,
                name,
                &r,
                "state.running_mean",
                model.running_mean,
            )?;
            check(
                &mut checks,
                name,
                &r,
                "state.running_var",
                model.running_var,
            )?;
            y
        } else {
            group_norm(
                x.clone().reshape([b, c, h * w]),
                weight.clone(),
                bias.clone(),
                2,
                1e-5,
            )
            .reshape([b, c, h, w])
        };
        check(&mut checks, name, &r, "output", y.clone())?;
        let grads = (y * r.tensor("upstream_gradient", &d)).sum().backward();
        check(
            &mut checks,
            name,
            &r,
            "input_gradient.0",
            x.grad(&grads).unwrap(),
        )?;
        check(
            &mut checks,
            name,
            &r,
            "gradient.weight",
            weight.grad(&grads).unwrap(),
        )?;
        check(
            &mut checks,
            name,
            &r,
            "gradient.bias",
            bias.grad(&grads).unwrap(),
        )?;
    }
    let mut bn = None;
    for name in [
        "batch_norm_train",
        "batch_norm_train_second",
        "batch_norm_eval",
    ] {
        let r = Reference::load(reference, name)?;
        let train = name != "batch_norm_eval";
        let model = bn.get_or_insert_with(|| BatchNorm {
            weight: r.tensor("state.weight", &d).require_grad(),
            bias: r.tensor("state.bias", &d).require_grad(),
            running_mean: Tensor::zeros([4], &d),
            running_var: Tensor::ones([4], &d),
            num_batches_tracked: 0,
            momentum: 0.1,
            epsilon: 1e-5,
        });
        let x = r
            .tensor::<4>(if train { "input.0" } else { "input" }, &d)
            .require_grad();
        let y = model.forward(x.clone(), train);
        check(&mut checks, name, &r, "output", y.clone())?;
        if train {
            let objective = (y * r.tensor("upstream_gradient", &d)).sum();
            check(&mut checks, name, &r, "objective", objective.clone())?;
            let grads = objective.backward();
            check(
                &mut checks,
                name,
                &r,
                "input_gradient.0",
                x.grad(&grads).unwrap(),
            )?;
            check(
                &mut checks,
                name,
                &r,
                "gradient.weight",
                model.weight.grad(&grads).unwrap(),
            )?;
            check(
                &mut checks,
                name,
                &r,
                "gradient.bias",
                model.bias.grad(&grads).unwrap(),
            )?;
        }
        check(
            &mut checks,
            name,
            &r,
            "state.running_mean",
            model.running_mean.clone(),
        )?;
        check(
            &mut checks,
            name,
            &r,
            "state.running_var",
            model.running_var.clone(),
        )?;
        check(
            &mut checks,
            name,
            &r,
            "state.num_batches_tracked",
            Tensor::<1>::ones([1], &d) * model.num_batches_tracked as f32,
        )?;
    }
    for name in ["group_norm", "group_norm_eval"] {
        let r = Reference::load(reference, name)?;
        let train = name == "group_norm";
        let x = r
            .tensor::<4>(if train { "input.0" } else { "input" }, &d)
            .require_grad();
        let shape = x.dims();
        let [b, c, h, w] = shape;
        let weight = r.tensor::<1>("state.weight", &d).require_grad();
        let bias = r.tensor::<1>("state.bias", &d).require_grad();
        let y = group_norm(
            x.clone().reshape([b, c, h * w]),
            weight.clone(),
            bias.clone(),
            2,
            1e-5,
        )
        .reshape(shape);
        check(&mut checks, name, &r, "output", y.clone())?;
        if train {
            let grads = (y * r.tensor("upstream_gradient", &d)).sum().backward();
            check(
                &mut checks,
                name,
                &r,
                "input_gradient.0",
                x.grad(&grads).unwrap(),
            )?;
            check(
                &mut checks,
                name,
                &r,
                "gradient.weight",
                weight.grad(&grads).unwrap(),
            )?;
            check(
                &mut checks,
                name,
                &r,
                "gradient.bias",
                bias.grad(&grads).unwrap(),
            )?;
        }
    }
    macro_rules! conv_check {
        ($name:literal,$rank:literal,$op:ident,$stride:expr,$padding:expr,$dilation:expr,$groups:expr) => {{
            let name = $name;
            let r = Reference::load(reference, name)?;
            let x = r.tensor::<$rank>("input.0", &d).require_grad();
            let weight = r.tensor::<$rank>("state.weight", &d).require_grad();
            let bias = r.tensor::<1>("state.bias", &d).require_grad();
            let y = $op(
                x.clone(),
                weight.clone(),
                Some(bias.clone()),
                ConvOptions::new($stride, $padding, $dilation, $groups),
            );
            check(&mut checks, name, &r, "output", y.clone())?;
            let grads = (y * r.tensor("upstream_gradient", &d)).sum().backward();
            check(
                &mut checks,
                name,
                &r,
                "input_gradient.0",
                x.grad(&grads).unwrap(),
            )?;
            check(
                &mut checks,
                name,
                &r,
                "gradient.weight",
                weight.grad(&grads).unwrap(),
            )?;
            check(
                &mut checks,
                name,
                &r,
                "gradient.bias",
                bias.grad(&grads).unwrap(),
            )?;
        }};
    }
    conv_check!("depthwise_dilated_conv1d", 3, conv1d, [1], [4], [2], 4);
    conv_check!("group_dilated_conv2d", 4, conv2d, [1, 1], [2, 2], [2, 2], 2);
    let name = "avg_pool_include_padding";
    let r = Reference::load(reference, name)?;
    let x = r.tensor::<4>("input.0", &d).require_grad();
    let y = avg_pool2d(
        x.clone(),
        AvgPoolOptions::new([5, 5])
            .with_stride([1, 1])
            .with_padding([2, 2])
            .with_count_include_pad(true),
    );
    check(&mut checks, name, &r, "output", y.clone())?;
    let grads = (y * r.tensor("upstream_gradient", &d)).sum().backward();
    check(
        &mut checks,
        name,
        &r,
        "input_gradient.0",
        x.grad(&grads).unwrap(),
    )?;
    let name = "spatial_audio_attention";
    let r = Reference::load(reference, name)?;
    let parameters: BTreeMap<_, _> = r
        .values
        .keys()
        .filter_map(|k| {
            k.strip_prefix("state.")
                .map(|name| (name.to_string(), r.flat(k, &d).require_grad()))
        })
        .collect();
    let model = SpatialAudioAttention {
        parameters,
        heads: 2,
        attention_dim: 8,
    };
    let x = r.tensor::<4>("input.0", &d).require_grad();
    let audio = r.tensor::<3>("input.1", &d).require_grad();
    let y = model.forward(x.clone(), audio.clone());
    check(&mut checks, name, &r, "output", y.clone())?;
    let grads = (y * r.tensor("upstream_gradient", &d)).sum().backward();
    check(
        &mut checks,
        name,
        &r,
        "input_gradient.0",
        x.grad(&grads).unwrap(),
    )?;
    check(
        &mut checks,
        name,
        &r,
        "input_gradient.1",
        audio.grad(&grads).unwrap(),
    )?;
    for (key, value) in &model.parameters {
        check(
            &mut checks,
            name,
            &r,
            &format!("gradient.{key}"),
            value.grad(&grads).unwrap(),
        )?;
    }
    Ok(OperatorReport {
        device: device.descriptor.clone(),
        checks,
    })
}
