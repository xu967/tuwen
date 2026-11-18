import torch

from ultralytics.models.yolovo.relate.local_tokens import PairLocalTokenExtractor


def _coordinate_scene(height, width):
    y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


def test_local_tokens_shapes_positions_and_coordinate_sampling():
    scene = _coordinate_scene(8, 8)
    subject_boxes = torch.tensor([[[0.25, 0.25, 0.5, 0.5]]])
    object_boxes = torch.tensor([[[0.75, 0.75, 0.5, 0.5]]])

    result = PairLocalTokenExtractor(grid_size=2)(scene, subject_boxes, object_boxes)

    assert result.subject.shape == (1, 1, 4, 2)
    assert result.object.shape == (1, 1, 4, 2)
    assert result.subject_positions.shape == (1, 1, 4, 2)
    assert result.subject_valid.shape == (1, 1, 4)
    torch.testing.assert_close(result.subject[0, 0], result.subject_positions[0, 0])
    torch.testing.assert_close(result.object[0, 0], result.object_positions[0, 0])
    assert result.subject_valid.all()
    assert result.object_valid.all()


def test_local_tokens_mask_canvas_degenerate_boxes_and_padded_pairs():
    scene = _coordinate_scene(8, 8)
    subject_boxes = torch.tensor([
        [[0.4, 0.5, 0.4, 0.4], [0.7, 0.5, 0.0, 0.4], [0.3, 0.5, 0.2, 0.2]],
    ])
    object_boxes = torch.tensor([
        [[0.8, 0.5, 0.2, 0.2], [0.2, 0.5, 0.2, 0.2], [0.2, 0.5, 0.2, 0.2]],
    ])
    valid_canvas = torch.tensor([[0.0, 0.0, 0.4, 1.0]])
    pair_valid = torch.tensor([[True, True, False]])

    result = PairLocalTokenExtractor(grid_size=2)(
        scene, subject_boxes, object_boxes, valid_canvas=valid_canvas, pair_valid=pair_valid
    )

    assert result.subject_valid[0, 0].tolist() == [True, False, True, False]
    assert not result.object_valid[0, 0].any()
    assert not result.subject_valid[0, 1].any()
    assert result.object_valid[0, 1].all()
    assert not result.subject_valid[0, 2].any()
    assert not result.object_valid[0, 2].any()
    assert torch.count_nonzero(result.subject[~result.subject_valid]) == 0
    assert torch.count_nonzero(result.object[~result.object_valid]) == 0


def test_local_tokens_empty_pairs_and_scene_gradients():
    extractor = PairLocalTokenExtractor(grid_size=2)
    empty_boxes = torch.empty(2, 0, 4)
    empty = extractor(torch.randn(2, 3, 5, 7), empty_boxes, empty_boxes)
    assert empty.subject.shape == (2, 0, 4, 3)

    scene = torch.randn(1, 3, 5, 7, requires_grad=True)
    boxes = torch.tensor([[[0.5, 0.5, 0.6, 0.6]]])
    result = extractor(scene, boxes, boxes)
    result.subject.sum().backward()
    assert scene.grad is not None
    assert scene.grad.abs().sum() > 0


def test_local_tokens_reject_malformed_inputs():
    extractor = PairLocalTokenExtractor()
    scene = torch.zeros(1, 3, 4, 4)
    boxes = torch.zeros(1, 1, 4)

    for invalid_call in (
        lambda: extractor(scene[0], boxes, boxes),
        lambda: extractor(scene, boxes[..., :3], boxes[..., :3]),
        lambda: extractor(scene, boxes, boxes, valid_canvas=torch.zeros(1, 1, 4)),
        lambda: extractor(scene, boxes, boxes, pair_valid=torch.ones(1, 2)),
    ):
        try:
            invalid_call()
        except ValueError:
            continue
        raise AssertionError("malformed input should raise ValueError")
