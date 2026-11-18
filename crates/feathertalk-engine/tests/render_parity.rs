use feathertalk_engine::inference::renderer::FramePicker;
#[test]
fn source_paste_area_center_truncate_linear() {
    use feathertalk_engine::{
        data::landmarks::{BgrFrame, Landmarks},
        inference::renderer::compose,
    };
    for size in [151, 152, 161, 203] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../.artifacts/render-reference/render_{size}.safetensors"
        ));
        let bytes = std::fs::read(path).unwrap();
        let safe = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let t = safe.tensor("image").unwrap();
        let frame = BgrFrame {
            width: t.shape()[1],
            height: t.shape()[0],
            stride: t.shape()[1] * 3,
            data: t.data().to_vec(),
        };
        let mut points = [[0; 2]; 110];
        points[1][0] = 8;
        points[31][0] = 8 + size;
        points[52][1] = 12;
        let prediction = safe
            .tensor("prediction")
            .unwrap()
            .data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        let actual = compose(&frame, &Landmarks { points }, &prediction).unwrap();
        let prepared =
            feathertalk_engine::data::geometry::prepare_frame(&frame, &Landmarks { points })
                .unwrap();
        let input = safe
            .tensor("input")
            .unwrap()
            .data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!([prepared.target, prepared.masked].concat(), input);
        assert_eq!(
            actual.data,
            safe.tensor("final").unwrap().data(),
            "size={size}"
        );
        assert_eq!(frame.data, t.data());
    }
}
#[test]
fn audio_windows_follow_output_timeline_across_resource_bounces() {
    for count in [1, 2, 24] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../.artifacts/render-reference/windows_{count}.safetensors"
        ));
        let bytes = std::fs::read(path).unwrap();
        let safe = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let floats = |name| {
            safe.tensor(name)
                .unwrap()
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let features = floats("features");
        let windows = floats("windows");
        let mut picker = FramePicker::new(2).unwrap();
        for k in 0..count {
            assert_eq!(picker.next(), 1 - k % 2);
            assert_eq!(
                feathertalk_engine::data::geometry::audio_window(&features, k, 0, count).unwrap(),
                windows[k * 40960..(k + 1) * 40960]
            );
        }
    }
}
#[test]
fn source_picker_bounces_without_repeated_endpoints() {
    for (n, expected) in [
        (2, vec![1, 0, 1, 0, 1]),
        (3, vec![1, 2, 1, 0, 1, 2, 1, 0, 1]),
        (4, vec![1, 2, 3, 2, 1, 0, 1, 2, 3]),
    ] {
        let mut p = FramePicker::new(n).unwrap();
        assert_eq!(
            (0..expected.len()).map(|_| p.next()).collect::<Vec<_>>(),
            expected
        );
        p.reset();
        assert_eq!(p.next(), 1);
    }
    assert!(FramePicker::new(0).is_err());
    assert!(FramePicker::new(1).is_err());
}
