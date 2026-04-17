from __future__ import annotations

import unittest

import torch

from omen_saliency import CANONICAL_ROLE_NAMES, SaliencyTraceModule
from omen_scale_config import OMENScaleConfig


class SaliencySemanticRolesTest(unittest.TestCase):
    def test_saliency_uses_canonical_role_names_and_named_facts(self) -> None:
        cfg = OMENScaleConfig.demo()
        mod = SaliencyTraceModule(d_tok=8, d_latent=16, cfg=cfg)

        self.assertEqual(mod.role_names[:6], CANONICAL_ROLE_NAMES)
        self.assertEqual(set(mod.named_role_predicates), set(CANONICAL_ROLE_NAMES))

        with torch.no_grad():
            mod.role_classifier.weight.zero_()
            mod.role_classifier.bias.zero_()
            for ridx in range(6):
                mod.role_classifier.weight[ridx, ridx] = 6.0

        token_hidden = torch.zeros(1, 6, 8)
        for token_idx in range(6):
            token_hidden[0, token_idx, token_idx] = 1.0
        z_neural = torch.randn(1, 16)
        attn = torch.zeros(1, 2, 2, 6, 6)

        out = mod(attn, token_hidden, z_neural, prover=None, train_step=0)
        self.assertEqual(out.sal_role_names[:6], CANONICAL_ROLE_NAMES)

        semantic_facts = out.sal_semantic_facts[0]
        semantic_preds = {fact.pred for fact in semantic_facts}
        for role_name, pred_id in mod.named_role_predicates.items():
            self.assertIn(pred_id, semantic_preds, role_name)

        unary_named = [fact for fact in semantic_facts if fact.pred in mod.named_role_predicates.values()]
        self.assertEqual(len(unary_named), 6)
        self.assertTrue(all(len(fact.args) == 1 for fact in unary_named))


if __name__ == "__main__":
    unittest.main()
