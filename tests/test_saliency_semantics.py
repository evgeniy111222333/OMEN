from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_saliency import CANONICAL_ROLE_NAMES, CANONICAL_ROLE_ONTOLOGY, SaliencyTraceModule
from omen_scale_config import OMENScaleConfig


class SaliencySemanticRolesTest(unittest.TestCase):
    def test_saliency_uses_canonical_role_names_and_named_facts(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.saliency_role_slots = len(CANONICAL_ROLE_ONTOLOGY)
        mod = SaliencyTraceModule(d_tok=len(CANONICAL_ROLE_ONTOLOGY), d_latent=16, cfg=cfg)

        self.assertEqual(mod.role_names[:6], CANONICAL_ROLE_NAMES)
        self.assertEqual(mod.role_names, CANONICAL_ROLE_ONTOLOGY)
        self.assertTrue(all(not name.startswith("role_") for name in mod.role_names))
        self.assertEqual(set(mod.named_role_predicates), set(CANONICAL_ROLE_ONTOLOGY))
        self.assertEqual(set(mod.named_role_relation_predicates), set(CANONICAL_ROLE_ONTOLOGY))

        with torch.no_grad():
            mod.role_classifier.weight.zero_()
            mod.role_classifier.bias.zero_()
            for ridx in range(mod.n_roles):
                mod.role_classifier.weight[ridx, ridx] = 6.0

        token_hidden = torch.zeros(1, 4, len(CANONICAL_ROLE_ONTOLOGY))
        action_idx = mod.role_to_idx["action"]
        agent_idx = mod.role_to_idx["agent"]
        patient_idx = mod.role_to_idx["patient"]
        temporal_idx = mod.role_to_idx["temporal"]
        token_hidden[0, 0, action_idx] = 1.0
        token_hidden[0, 1, agent_idx] = 1.0
        token_hidden[0, 2, patient_idx] = 1.0
        token_hidden[0, 3, temporal_idx] = 1.0
        z_neural = torch.randn(1, 16)
        attn = torch.zeros(1, 2, 2, 4, 4)

        attn[:, :, :, 1, 0] = 0.95
        attn[:, :, :, 2, 0] = 0.95
        attn[:, :, :, 3, 0] = 0.95

        out = mod(attn, token_hidden, z_neural, prover=None, train_step=0)
        self.assertEqual(out.sal_role_names, CANONICAL_ROLE_ONTOLOGY)

        semantic_facts = out.sal_semantic_facts[0]
        semantic_preds = {fact.pred for fact in semantic_facts}
        for role_name, pred_id in mod.named_role_predicates.items():
            if role_name in {"action", "agent", "patient", "temporal"}:
                self.assertIn(pred_id, semantic_preds, role_name)

        unary_named = [fact for fact in semantic_facts if fact.pred in mod.named_role_predicates.values()]
        self.assertGreaterEqual(len(unary_named), 4)
        self.assertTrue(all(len(fact.args) == 1 for fact in unary_named))

        binary_named = [fact for fact in semantic_facts if fact.pred in mod.named_role_relation_predicates.values()]
        self.assertTrue(any(fact.pred == mod.named_role_relation_predicates["agent"] for fact in binary_named))
        self.assertTrue(any(fact.pred == mod.named_role_relation_predicates["patient"] for fact in binary_named))
        self.assertTrue(any(fact.pred == mod.named_role_relation_predicates["temporal"] for fact in binary_named))
        self.assertTrue(all(len(fact.args) == 2 for fact in binary_named))

    def test_default_canonical_prefix_stays_stable(self) -> None:
        cfg = OMENScaleConfig.demo()
        mod = SaliencyTraceModule(d_tok=8, d_latent=16, cfg=cfg)
        self.assertEqual(mod.role_names[:6], CANONICAL_ROLE_NAMES)


if __name__ == "__main__":
    unittest.main()
