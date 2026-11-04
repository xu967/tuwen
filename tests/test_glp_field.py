"""GLP field unit tests (V1 absolute + V2 subject-frame). CPU only."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.models.yolovo.relate.field import (
    FieldConfig, PredicateFieldRead, compose_fields, field_grid, field_stats,
    gate, to_image_frame,
)
from ultralytics.models.yolovo.relate.head import YOLOVORelate, _VocabHead


def _tensors(B=1, K=3, M=2, d=32, A=4, G=2, hw=(8, 8)):
    torch.manual_seed(0)
    queries = torch.randn(B, K, d)
    scene = torch.randn(B, d, *hw)
    anchors = torch.rand(B, K, A, 4) * 0.6 + 0.2
    anchors[..., 2:] = anchors[..., 2:].clamp_min(0.1)
    geo = torch.randn(B, K, 19)
    e = torch.randn(B, K, M, d)
    return queries, scene, anchors, geo, e


def test_identity_init_v1():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2)
    f = PredicateFieldRead(d_model=32, cfg=cfg)
    q, scene, anchors, geo, e = _tensors()
    out, field = f(q, scene, anchors, geo, e, return_field=True)
    assert torch.allclose(out, q.unsqueeze(2), atol=1e-6)
    assert field["mu"].shape == (1, 3, 2, 8, 2)
    assert torch.isfinite(field["log_omega"]).all()


def test_identity_init_v2():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2,
                      local_frame=True, use_null_bin=True, use_geo=False)
    f = PredicateFieldRead(d_model=32, cfg=cfg)
    q, scene, anchors, geo, e = _tensors()
    out, field = f(q, scene, anchors, geo, e, return_field=True)
    assert torch.allclose(out, q.unsqueeze(2), atol=1e-6)
    w = field["logw"].exp()
    assert torch.allclose(w.sum(-1) + field["w_bg"], torch.ones_like(field["w_bg"]),
                          atol=1e-5)


def test_v2_object_independence_and_v1_dependence():
    q, scene, anchors, geo, e = _tensors()
    anchors2, geo2 = anchors.clone(), geo.clone()
    anchors2[:, :, 1, 0] += 0.2          # move the object box only
    geo2[:, :, 0] += 1.0                 # and change object geometry

    v2 = PredicateFieldRead(d_model=32, cfg=FieldConfig(
        mode="glp", n_candidates=2, n_kernels=2,
        local_frame=True, use_null_bin=True, use_geo=False))
    _, f1 = v2(q, scene, anchors, geo, e, return_field=True)
    _, f2 = v2(q, scene, anchors2, geo2, e, return_field=True)
    assert torch.allclose(f1["mu_local"], f2["mu_local"], atol=1e-6)
    assert torch.allclose(f1["logw"], f2["logw"], atol=1e-6)

    v1 = PredicateFieldRead(d_model=32, cfg=FieldConfig(
        mode="glp", n_candidates=2, n_kernels=2))
    _, g1 = v1(q, scene, anchors, geo, e, return_field=True)
    _, g2 = v1(q, scene, anchors2, geo2, e, return_field=True)
    assert not torch.allclose(g1["mu"], g2["mu"], atol=1e-6)


def test_null_bin_conservation_and_abstention():
    # a dominant negative beta drives spatial mass to zero
    logw = torch.full((1, 3, 2, 8), -20.0)
    beta = torch.full((1, 3, 2), 20.0)
    joint = F.log_softmax(torch.cat([logw, beta.unsqueeze(-1)], -1), dim=-1)
    assert joint[..., -1].exp().mean() > 0.999
    assert joint[..., :-1].exp().sum(-1).max() < 1e-6


def test_null_bin_attenuates_read():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2,
                      local_frame=True, use_null_bin=True, use_geo=False)
    f = PredicateFieldRead(d_model=32, cfg=cfg)
    torch.nn.init.normal_(f.read_out.weight, std=0.1)  # open the read-out
    q, scene, anchors, geo, e = _tensors()
    beta_idx = f.n_kernels * 5
    with torch.no_grad():
        f.hyper[-1].bias[beta_idx] = -10.0
        low, f_low = f(q, scene, anchors, geo, e, return_field=True)
        f.hyper[-1].bias[beta_idx] = 10.0
        high, f_high = f(q, scene, anchors, geo, e, return_field=True)
    assert float(f_high["w_bg"].mean()) > float(f_low["w_bg"].mean())
    assert high.norm() < low.norm()          # background leak attenuates the read


def test_eval_budget_honoured_in_both_stages():
    rel = _relate_head(FieldConfig(mode="glp", n_candidates=2, n_kernels=2))
    base = torch.randn(1, 32, 8, 8)
    boxes = torch.rand(1, 10, 4) * 0.5 + 0.25
    boxes[..., 2:] = boxes[..., 2:].clamp_min(0.05)
    with torch.no_grad():
        out = rel._compute(base, boxes, gt_list=None, eval_budget=20)
    assert out["sub_idx"].shape[0] == 20


def test_compose_drops_stale_metadata_and_negate():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2,
                      local_frame=True, use_null_bin=True, use_geo=False)
    f = PredicateFieldRead(d_model=32, cfg=cfg)
    q, scene, anchors, geo, e = _tensors()
    _, fa = f(q, scene, anchors, geo, e, return_field=True)
    fb = {k: v[0] for k, v in fa.items()}
    comp = compose_fields(fb, fb, "or")
    assert "w_bg" not in comp and "mu_local" not in comp and "subject_box" not in comp
    assert torch.allclose(comp["logw"].exp().sum(-1), torch.ones_like(comp["logw"][..., 0]),
                          atol=1e-5)
    one = {"mu": torch.zeros(1, 1, 2), "sigma": torch.ones(1, 1, 2),
           "logw": torch.zeros(1, 1)}
    neg = compose_fields(one, one, "negate")
    assert torch.allclose(neg["mass"], torch.zeros(1), atol=1e-5)


def test_translation_and_scale_equivariance():
    subject = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    mu_local = torch.tensor([[1.0, 0.0]])
    sig_local = torch.tensor([[0.1, 0.2]])
    mu, sig = to_image_frame(mu_local, sig_local, subject)
    assert torch.allclose(mu, torch.tensor([[0.7, 0.5]]), atol=1e-6)
    assert torch.allclose(sig, torch.tensor([[0.02, 0.08]]), atol=1e-6)

    t = torch.tensor([0.1, -0.05])
    s2 = subject.clone()
    s2[:, :2] += t
    mu2, _ = to_image_frame(mu_local, sig_local, s2)
    assert torch.allclose(mu2 - mu, t.unsqueeze(0), atol=1e-6)

    a = 1.5
    s3 = subject.clone()
    s3[:, 2:] *= a
    mu3, sig3 = to_image_frame(mu_local, sig_local, s3)
    assert torch.allclose(mu3 - subject[:, :2], a * (mu - subject[:, :2]), atol=1e-6)
    # covariance determinant scales by a^2 per axis -> a^4 in 2-D
    det1 = (sig[..., 0] * sig[..., 1])
    det3 = (sig3[..., 0] * sig3[..., 1])
    assert torch.allclose(det3 / det1, torch.tensor(a ** 2), atol=1e-5)
    # V2 must not clamp the absolute sigma: large local sigma still scales by a
    sigma_big = torch.tensor([[0.8, 0.8]])
    _, sb = to_image_frame(mu_local, sigma_big, subject)
    _, sb3 = to_image_frame(mu_local, sigma_big, s3)
    assert torch.allclose(sb3 / sb, torch.full((1, 2), a), atol=1e-5)


def test_gaussian_compose_and():
    a = {"mu": torch.tensor([[[0.0, 0.0]]]), "sigma": torch.tensor([[[0.5, 0.5]]]),
         "logw": torch.zeros(1, 1)}
    b = {"mu": torch.tensor([[[0.4, 0.0]]]), "sigma": torch.tensor([[[0.25, 0.25]]]),
         "logw": torch.zeros(1, 1)}
    out = compose_fields(a, b, "and")
    prec = 1 / 0.5 ** 2 + 1 / 0.25 ** 2
    mu = (0.0 / 0.5 ** 2 + 0.4 / 0.25 ** 2) / prec
    assert torch.allclose(out["mu"][0, 0], torch.tensor([mu, 0.0]), atol=1e-6)
    assert torch.allclose(out["sigma"][0, 0], torch.full((2,), prec ** -0.5), atol=1e-6)
    assert torch.allclose(out["logw"].exp().sum(-1), torch.ones(1), atol=1e-5)


def test_gaussian_compose_or_multi():
    a = {"mu": torch.tensor([[[0.0, 0.0], [0.1, 0.1]]]),
         "sigma": torch.full((1, 2, 2), 0.2), "logw": torch.log(torch.tensor([[0.5, 0.5]]))}
    b = {"mu": torch.tensor([[[0.3, 0.0], [0.4, 0.1], [0.5, 0.2]]]),
         "sigma": torch.full((1, 3, 2), 0.3), "logw": torch.log(torch.full((1, 3), 1 / 3))}
    out = compose_fields(a, b, "or")
    assert out["mu"].shape == (1, 3, 2)          # pruned to max(Ka, Kb) = 3
    assert torch.allclose(out["logw"].exp().sum(-1), torch.ones(1), atol=1e-5)


def test_gaussian_compose_and_multi():
    a = {"mu": torch.tensor([[[0.0, 0.0], [0.2, 0.0]]]),
         "sigma": torch.full((1, 2, 2), 0.5), "logw": torch.log(torch.tensor([[0.5, 0.5]]))}
    b = {"mu": torch.tensor([[[0.0, 0.0], [0.4, 0.0]]]),
         "sigma": torch.full((1, 2, 2), 0.25), "logw": torch.log(torch.tensor([[0.5, 0.5]]))}
    out = compose_fields(a, b, "and")
    assert out["mu"].shape == (1, 2, 2)          # pruned to max(Ka, Kb) = 2
    assert torch.allclose(out["logw"].exp().sum(-1), torch.ones(1), atol=1e-5)
    # strongest pair is (0,0): equal means -> mean 0, precision added
    assert torch.allclose(out["mu"][0, 0], torch.zeros(2), atol=1e-6)


def test_gaussian_compose_transitive():
    a = {"mu": torch.tensor([[[0.0, 0.0]]]), "sigma": torch.tensor([[[0.3, 0.3]]]),
         "logw": torch.zeros(1, 1)}
    b = {"mu": torch.tensor([[[0.2, 0.1]]]), "sigma": torch.tensor([[[0.4, 0.4]]]),
         "logw": torch.zeros(1, 1)}
    out = compose_fields(a, b, "transitive")
    assert torch.allclose(out["mu"][0, 0], torch.tensor([0.2, 0.1]), atol=1e-6)
    assert torch.allclose(out["sigma"][0, 0], torch.full((2,), 0.5), atol=1e-6)


def test_field_grid_and_stats():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2)
    f = PredicateFieldRead(d_model=32, cfg=cfg)
    q, scene, anchors, geo, e = _tensors()
    _, field = f(q, scene, anchors, geo, e, return_field=True)
    field1 = {k: (v[0] if v.dim() and v.shape[0] == 1 else v) for k, v in field.items()}
    g = field_grid(field1, k=0, m=0, hw=(16, 16))
    assert g.shape == (16, 16) and g.max() <= 1.0 + 1e-5 and (g >= 0).all()
    st = field_stats(field1, anchors[0])
    for key in ("mass_in_sub", "mass_in_obj", "mass_in_contact", "mass_outside",
                "peak", "kernel_entropy", "alignment"):
        assert key in st, key
    assert 0.0 <= float(st["mass_in_union"]) <= 1.0


def test_select_candidates_forces_gt():
    class _Stub:
        pass

    head = _Stub()
    logits = torch.randn(1, 3, 10)
    gtp = torch.tensor([[7, -1, 2]])
    cand = YOLOVORelate._select_candidates(head, logits, gtp,
                                           torch.ones(1, 3, dtype=torch.bool),
                                           m=3, training=True)
    assert cand.shape == (1, 3, 3)
    assert 7 in cand[0, 0].tolist()
    assert 2 in cand[0, 2].tolist()


def _relate_head(cfg):
    torch.manual_seed(0)
    rel = YOLOVORelate(
        primary_head=None, feat_layers=[0], feat_strides=[16], n_predicates=4,
        text_dim=32, geo_budget=8, final_budget=4, eval_budget=4,
        max_objects=10, backbone_dim=768, d_model=32, field_cfg=cfg)
    W = torch.randn(4, 32)
    rel.install_vocabulary(["on", "left", "right", "under"], W)
    return rel


def test_head_field_smoke_and_backward():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2)
    rel = _relate_head(cfg)
    base = torch.randn(1, 32, 8, 8)
    boxes = torch.rand(1, 8, 4) * 0.5 + 0.25
    boxes[..., 2:] = boxes[..., 2:].clamp_min(0.05)
    out = rel._compute(base, boxes, gt_list=[[(0, 1, 0), (2, 3, 1)]],
                       box_cls=torch.zeros(8, dtype=torch.long))
    assert "field" in out and "cand_idx" in out and "field_cos" in out
    assert out["cand_idx"].shape[0] == out["logits"].shape[0]
    loss = sum(v for k, v in out.items() if k in
               ("nce", "swap", "sigmoid", "obj", "bg", "geo", "rel"))
    loss.backward()
    assert rel.field_read.hyper[-1].weight.grad is not None
    assert torch.isfinite(loss)


def test_head_field_smoke_v2():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2,
                      local_frame=True, use_null_bin=True, use_geo=False)
    rel = _relate_head(cfg)
    base = torch.randn(1, 32, 8, 8)
    boxes = torch.rand(1, 8, 4) * 0.5 + 0.25
    boxes[..., 2:] = boxes[..., 2:].clamp_min(0.05)
    out = rel._compute(base, boxes, gt_list=[[(0, 1, 0), (2, 3, 1)]],
                       box_cls=torch.zeros(8, dtype=torch.long))
    assert "w_bg" in out["field"]
    assert out["field"]["mu_local"].shape[-1] == 2
    loss = sum(v for k, v in out.items() if k in
               ("nce", "swap", "sigmoid", "obj", "bg", "geo", "rel"))
    loss.backward()
    assert rel.field_read.hyper[-1].weight.grad is not None
    assert torch.isfinite(loss)


def test_field_purity_v2():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2,
                      local_frame=True, use_null_bin=True, use_geo=False)
    f = PredicateFieldRead(d_model=32, cfg=cfg)
    q, scene, anchors, geo, e = _tensors()
    out, field = f(q, scene, anchors, geo, e, return_field=True)
    # w_agg must be exactly the spatial slice of softmax([logw; beta])
    logw = field["logw"]
    wbg = field["w_bg"]
    assert torch.allclose(logw.exp().sum(-1) + wbg, torch.ones_like(wbg), atol=1e-5)
    assert torch.allclose(field["w_agg"], logw.exp(), atol=1e-6)
    assert torch.allclose(out, q.unsqueeze(2), atol=1e-6)  # read-out is zero -> identity


def test_score_all_returns_field():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2)
    rel = _relate_head(cfg)
    rel.eval()
    rel.capture = True
    base = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        rel([None, base])
        boxes = torch.rand(8, 4) * 0.5 + 0.25
        boxes[..., 2:] = boxes[..., 2:].clamp_min(0.05)
        sub, obj, valid, logits, rel_logits, field, cand = rel.score_all(
            boxes, topk=4, return_field=True)
    assert field is not None and cand is not None
    assert sub.shape == obj.shape == valid.shape
    assert logits.shape[0] == sub.shape[0]


def test_gate_reweights_kernels():
    fp = {"logw": torch.log(torch.full((3,), 1 / 3)),
          "log_omega": torch.tensor([0.0, 1.0, 2.0])}
    fr = {"log_omega": torch.tensor([-1.0, 0.0, 1.0])}
    out = gate(fp, fr, tau=1.0)
    assert torch.allclose(out["logw"].exp().sum(), torch.tensor(1.0), atol=1e-5)
    assert int(out["logw"].argmax()) == 2


def test_field_metrics():
    from ultralytics.models.yolovo.relate.field_metrics import (
        aurc, gaussian_box_mass, subject_frame_consistency,
    )
    # centred narrow Gaussian: almost all mass inside the box
    mu = torch.full((1, 1, 2), 0.5)
    sig = torch.full((1, 1, 2), 0.01)
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    assert float(gaussian_box_mass(mu, sig, box)) > 0.99
    far = torch.tensor([[0.0, 0.0, 0.01, 0.01]])
    assert float(gaussian_box_mass(mu, sig, far)) < 1e-3
    # perfect confidence -> zero AURC
    assert aurc(torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 1, 1])) == 0.0

    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2,
                      local_frame=True, use_null_bin=True)
    f = PredicateFieldRead(d_model=32, cfg=cfg)
    q, scene, anchors, geo, e = _tensors()
    sfc = subject_frame_consistency(f, q, scene, anchors, geo, e)
    assert sfc["field_sfc_mu"] < 1e-4 and sfc["field_sfc_sigma"] < 1e-4
    assert sfc["field_ee"] < 1e-4


def test_score_all_empty_boxes():
    cfg = FieldConfig(mode="glp", n_candidates=2, n_kernels=2)
    rel = _relate_head(cfg)
    rel.capture = True
    rel._captured = {"feats": [torch.randn(1, 32, 8, 8)]}
    sub, obj, valid, logits, rel_logits, field, cand = rel.score_all(
        torch.zeros(1, 4), return_field=True)
    assert sub.numel() == obj.numel() == valid.numel() == 0
    assert logits.shape == (0, 4)
    assert field is None and cand is None


def test_field_metric_accumulator_update():
    from ultralytics.models.yolovo.relate.field_metrics import FieldMetricAccumulator
    K, M, AG = 2, 2, 2
    out = {
        "sub_idx": torch.tensor([0, 2]),
        "obj_idx": torch.tensor([1, 3]),
        "valid_mask": torch.tensor([True, True]),
        "logits": torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
        "rel_logits": torch.tensor([1.0, -1.0]),
        "boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1],
                                [0.2, 0.6, 0.1, 0.1],
                                [0.6, 0.2, 0.1, 0.1],
                                [0.6, 0.6, 0.1, 0.1]]]),
        "cand_idx": torch.tensor([[0, 1], [1, 2]]),
        "field": {
            "mu": torch.full((K, M, AG, 2), 0.2),
            "sigma": torch.full((K, M, AG, 2), 0.05),
            "logw": torch.log(torch.full((K, M, AG), 1.0 / AG)),
            "w_bg": torch.tensor([[0.1, 0.4], [0.2, 0.5]]),
        },
    }
    acc = FieldMetricAccumulator(4)
    acc.update(out, [[(0, 1, 0)]])
    m = acc.compute()
    assert "rel_top1" in m and "rel_exist_f1" in m
    assert "field_fma" in m and "field_wbg_gap" in m
    assert m["field_wbg_gap"] > 0.0     # GT w_bg (0.1) < non-GT (0.4)
    assert m["rel_top1"] >= 0.0


def test_score_candidates_matches_full():
    vh = _VocabHead(d_model=16, text_dim=8)
    torch.manual_seed(1)
    W = torch.randn(5, 8)
    vh.set_vocabulary([str(i) for i in range(5)], W)
    q = F.normalize(torch.randn(1, 2, 3, 8), dim=-1)
    qs = F.normalize(torch.randn(1, 2, 3, 8), dim=-1)
    cand = torch.tensor([[[0, 2, 4], [1, 3, 4]]])
    cos, logit = vh.score_candidates(q, qs, cand)
    Wc = vh.W[cand]
    a = vh.alpha[cand]
    exp_cos = (1 - a) * (q * Wc).sum(-1) + a * (qs * Wc).sum(-1)
    assert torch.allclose(cos, exp_cos, atol=1e-5)
    assert torch.allclose(logit, exp_cos * vh.logit_scale.exp() + vh.logit_bias,
                          atol=1e-5)


def test_batched_forward_matches_per_image_losses():
    """Padded whole-batch forward must give the same per-image losses."""
    rel = _relate_head(FieldConfig(mode="glp", n_candidates=2, n_kernels=2))
    rel.eval()  # 关掉 dropout，比较的是数值路径而不是随机 mask
    torch.manual_seed(0)
    base = torch.randn(2, 768, 4, 4)
    boxes = torch.tensor([
        [[0.3, 0.4, 0.2, 0.2], [0.6, 0.5, 0.3, 0.2], [0.5, 0.3, 0.1, 0.1]],
        [[0.4, 0.4, 0.2, 0.2], [0.7, 0.6, 0.2, 0.2], [0.5, 0.5, 0.01, 0.01]],
    ])
    counts = torch.tensor([3, 2])
    gt = [[(0, 1, 0), (1, 2, 1)], [(0, 1, 2)]]
    cls = [torch.tensor([0, 1, 2]), torch.tensor([0, 1])]
    o = rel._forward_pairs(base, boxes, counts, gt_list=gt)
    assert o["sub_idx"].shape[0] == 2 and o["logits"].shape[0] == 2
    for i in range(2):
        out_i = {
            "logits": o["logits"][i], "sub_idx": o["sub_idx"][i],
            "obj_idx": o["obj_idx"][i], "valid_mask": o["valid_mask"][i],
            "rel_logits": o["rel_logits"][i], "boxes": boxes[i:i + 1],
            "cand_idx": o["cand_idx"][i], "field_cos": o["field_cos"][i],
            "field": {k: v[i] for k, v in o["field"].items()},
        }
        if "logits_raw" in o:
            out_i["logits_raw"] = o["logits_raw"][i]
        batched = rel._losses(
            out_i, [gt[i]], o["q"][i:i + 1], o["is_gt"][i:i + 1],
            o["sub_idx"][i:i + 1], o["obj_idx"][i:i + 1], o["valid_mask"][i:i + 1],
            o["v_obj"][i:i + 1], boxes[i:i + 1], o["n_boxes"], cls[i],
            box_counts=counts[i:i + 1])
        n = int(counts[i])
        single = rel._compute(base[i:i + 1], boxes[i:i + 1, :n],
                              gt_list=[gt[i]], box_cls=cls[i].unsqueeze(0))
        for k in ("nce", "swap", "sigmoid", "obj", "bg", "geo", "rel"):
            assert abs(float(batched[k]) - float(single[k])) < 1e-4, (i, k, batched[k], single[k])
