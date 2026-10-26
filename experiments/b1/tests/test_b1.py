import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE / 'engine'))
from trust_quality import crossfit, grouped_folds, make_targets
from trust_training import (classification_loss, consistency_loss,
                            configure_parameters, freeze_warmup_gradients, load_quality)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.vision_model = nn.Module()
        self.backbone.vision_model.encoder = nn.Module()
        self.backbone.vision_model.encoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.backbone.vision_model.post_layernorm = nn.LayerNorm(4)
        self.backbone.visual_projection = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 2, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.ones(()))

    def forward(self, x):
        for layer in self.backbone.vision_model.encoder.layers:
            x = layer(x).tanh()
        x = self.backbone.vision_model.post_layernorm(x) * self.gate
        return self.head(self.backbone.visual_projection(x)) * self.log_scale.exp()


class B1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.cfg = json.loads((CODE / 'quality_config.json').read_text())

    def test_duplicate_groups_never_cross_folds(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array(['a', 'a', 'b', 'b', 'c', 'd', 'e', 'f'])
        folds = grouped_folds(y, groups, 3, 42)
        for g in np.unique(groups):
            self.assertEqual(len(set(folds[groups == g])), 1)
        np.testing.assert_array_equal(folds, grouped_folds(y, groups, 3, 42))

    def test_quality_binding_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'cache').mkdir()
            (root / 'engine').mkdir()
            labels = root / 'engine/train_labels.npy'
            np.save(labels, np.array([0, 1]))
            path = root / 'cache/train_quality.npz'
            np.savez(path, targets=np.eye(750, dtype=np.float32)[:2],
                     trust=np.ones(2, np.float32), group_weight=np.ones(2, np.float32))
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            path.with_suffix('.json').write_text(json.dumps({
                'quality_sha256': digest(path),
                'inputs': {'engine/train_labels.npy': digest(labels)}}))
            arrays, _ = load_quality(root, digest, 'cpu')
            self.assertEqual(tuple(arrays['targets'].shape), (2, 750))
            np.save(labels, np.array([1, 0]))
            with self.assertRaisesRegex(ValueError, 'input changed'):
                load_quality(root, digest, 'cpu')

    def test_missing_class_is_reported_without_nan(self):
        x = np.eye(4, dtype=np.float32)
        y = np.array([0, 0, 0, 1])
        folds = np.array([0, 1, 2, 0])
        with patch('trust_quality.grouped_folds', return_value=folds):
            p, v, _, missing = crossfit(x, y, np.arange(4), self.cfg)
        self.assertIn(1, missing[0])
        self.assertTrue(np.isfinite(p).all() and np.isfinite(v).all())

    def test_oof_predictions_do_not_read_heldout_labels(self):
        x = np.random.default_rng(4).normal(size=(12, 4)).astype('float32')
        y = np.tile([0, 1], 6)
        groups = np.arange(12)
        folds = np.repeat([0, 1, 2], 4)
        # Freeze the split, then change only held-out labels. The classifier AND
        # neighbor outputs for that fold must be unaffected.
        with patch('trust_quality.grouped_folds', return_value=folds):
            p1, v1, _, _ = crossfit(x, y, groups, self.cfg)
            y2 = y.copy()
            y2[:4] = 1 - y2[:4]
            p2, v2, _, _ = crossfit(x, y2, groups, self.cfg)
        np.testing.assert_allclose(p1[:4], p2[:4])
        np.testing.assert_allclose(v1[:4], v2[:4])
        np.testing.assert_allclose(p1.sum(1), 1, atol=1e-6)

    def test_duplicate_weight_and_conservative_correction(self):
        p = np.array([[.05, .95], [.05, .95], [.9, .1]], np.float32)
        v = p.copy()
        targets, trust, weights = make_targets(p, v, np.array([0, 0, 1]), np.array([0, 0, 1]), self.cfg)
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertTrue((trust >= .1).all())
        self.assertGreater(targets[0, 1], 0)
        self.assertGreater(targets[0, 0], 0)
        v[0] = [.99, .01]  # disagreement prevents correction
        targets, _, _ = make_targets(p, v, np.array([0, 0, 1]), np.array([0, 0, 1]), self.cfg)
        np.testing.assert_array_equal(targets[0], [1, 0])

    def test_mixed_loss_keeps_each_samples_weight(self):
        logits = torch.tensor([[2., -1.], [-.5, 1.]], requires_grad=True)
        target = torch.eye(2)
        trust = torch.tensor([.2, 1.])
        weight = torch.tensor([1.5, .5])
        perm = torch.tensor([1, 0])
        mixed = classification_loss(logits, target, trust, weight, .7, .025, perm, .3)
        a = classification_loss(logits, target, trust, weight, .7, .025)
        b = classification_loss(logits, target[perm], trust[perm], weight[perm], .7, .025)
        torch.testing.assert_close(mixed, .3 * a + .7 * b)
        mixed.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_consistency_has_gradients_on_both_views(self):
        a = torch.tensor([[2., 0.]], requires_grad=True)
        b = torch.tensor([[0., 2.]], requires_grad=True)
        loss = consistency_loss(a, b, torch.tensor([.1]), torch.ones(1))
        loss.backward()
        self.assertGreater(float(a.grad.abs().sum()), 0)
        self.assertGreater(float(b.grad.abs().sum()), 0)
        self.assertAlmostEqual(float(consistency_loss(a, a, torch.zeros(1), torch.ones(1)).detach()), 0, places=6)

    def test_warmup_then_unfreeze_and_optimizer_resume(self):
        model = TinyModel()
        configure_parameters(model, 1)
        self.assertFalse(model.backbone.vision_model.encoder.layers[0].weight.requires_grad)
        self.assertFalse(model.backbone.visual_projection.weight.requires_grad)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        x = torch.randn(6, 4)
        target = torch.eye(2).repeat(3, 1)
        for step in [0, 1]:
            optimizer.zero_grad(set_to_none=True)
            a, b = model(x), model(x + .1)
            loss = classification_loss(a, target, torch.ones(6), torch.ones(6), .7, .025)
            loss += consistency_loss(a, b, torch.zeros(6), torch.ones(6))
            loss.backward()
            freeze_warmup_gradients(model, step, 1)
            optimizer.step()
            if step == 0:
                for name, p in model.named_parameters():
                    if not (name.startswith('head.') or name == 'log_scale'):
                        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                self.assertFalse(torch.equal(model.head.weight, before['head.weight']))
        self.assertFalse(torch.equal(model.backbone.vision_model.encoder.layers[-1].weight,
                                     before['backbone.vision_model.encoder.layers.2.weight']))
        clone = TinyModel()
        configure_parameters(clone, 1)
        clone.load_state_dict(model.state_dict())
        resumed = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad], lr=.01)
        resumed.load_state_dict(optimizer.state_dict())
        self.assertEqual(len(resumed.state), len(optimizer.state))


if __name__ == '__main__':
    unittest.main()
