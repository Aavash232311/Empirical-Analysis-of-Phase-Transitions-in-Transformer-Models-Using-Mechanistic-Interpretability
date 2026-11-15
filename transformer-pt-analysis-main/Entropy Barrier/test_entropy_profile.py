"""Run: python -m unittest -v test_entropy_profile.py"""
import math
import io
import unittest
import numpy as np
import torch
from entropy_profile import (ShellWangLandau, diagonal_mass, stable_ce,
                             reflected, estimate_barrier)


class EntropyTests(unittest.TestCase):
    def test_spectral_rule_and_gauge(self):
        p = 7
        a, b, c = torch.meshgrid(*(torch.arange(p, dtype=torch.float64),) * 3, indexing="ij")
        logits = torch.cos(2 * math.pi * (a + b - c) / p)
        self.assertAlmostEqual(diagonal_mass(logits).item(), 1, places=12)
        gauge = torch.randn(p, p, 1, dtype=torch.float64)
        self.assertAlmostEqual(diagonal_mass(3 * logits + gauge).item(), 1, places=12)
        off_diagonal = torch.cos(2 * math.pi * (a + 2 * b - c) / p)
        self.assertLess(diagonal_mass(off_diagonal).item(), 1e-20)
        with self.assertRaises(ValueError):
            diagonal_mass(torch.zeros(p, p, p))

    def test_small_ce_and_extra_class(self):
        z = torch.tensor([[100., 0., -10.]], dtype=torch.float64)
        loss = stable_ce(z, torch.tensor([0])).item()
        self.assertGreater(loss, 0)
        self.assertAlmostEqual(math.log(loss), math.log(math.exp(-100) + math.exp(-110)), places=12)
        z = torch.tensor([[1., 2., 20.]], dtype=torch.float64)
        self.assertAlmostEqual(stable_ce(z, torch.tensor([1])).item(),
                               torch.nn.functional.cross_entropy(z, torch.tensor([1])).item(), places=12)

    def test_reflection(self):
        v = np.array([-1001., -3., -1.2, 0., 1.2, 3., 1001.])
        r = reflected(v, 1)
        self.assertTrue((np.abs(r) <= 1).all())
        np.testing.assert_allclose(r, -reflected(-v, 1))

    def test_rejections_count_current_state(self):
        def evaluate(t):
            return (0., .25) if t[0] == 0 else (10., .75)
        s = ShellWangLandau(evaluate, torch.zeros(1), [0, .5, 1], 0, .1, 1, .1, 1, 4)
        for _ in range(10):
            self.assertFalse(s.move())
        np.testing.assert_array_equal(s.hist, [10, 0])
        self.assertEqual(s.theta.item(), 0)
        self.assertFalse(s.refine(.8, 1))

    def test_resume_exact(self):
        def evaluate(t):
            return 0., (t[0].item() + 1) / 2
        def make(seed):
            return ShellWangLandau(evaluate, torch.zeros(1, dtype=torch.float64), np.linspace(0, 1, 11), 0, .1, 1, .3, 1, seed)
        a, b = make(1), make(2)
        for _ in range(50):
            a.move()
        buffer = io.BytesIO()
        torch.save(a.state_dict(), buffer)
        buffer.seek(0)
        b.load_state(torch.load(buffer, weights_only=True))
        for _ in range(100):
            a.move()
            b.move()
        np.testing.assert_array_equal(a.bias, b.bias)
        self.assertTrue(torch.equal(a.theta, b.theta))
        self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)

    def test_barrier_and_missing_support(self):
        y = np.linspace(.05, .95, 10)
        entropy = (y - .5) ** 2 * 10
        counts = np.full(10, 100)
        result = estimate_barrier(y, entropy, counts, [0, .2], [.8, 1])
        self.assertEqual(result["status"], "candidate_projected_barrier")
        self.assertAlmostEqual(result["delta_s_left"], 2.)
        counts[4] = 0
        self.assertEqual(estimate_barrier(y, entropy, counts, [0, .2], [.8, 1])["status"], "insufficient_coverage")
        self.assertEqual(estimate_barrier(y, y, np.full(10, 100), [0, .2], [.8, 1])["status"], "no_resolved_interior_trough")

    def test_known_density_of_states(self):
        # For each y=(theta0+1)/2, shell admits |theta1| <= 0.4*w(y).
        # Its exact DOS is proportional to w(y)=0.2+0.8*(2y-1)^2.
        def evaluate(t):
            y = (t[0].item() + 1) / 2
            w = .2 + .8 * (2 * y - 1) ** 2
            return t[1].item() / w, y
        edges = np.linspace(0, 1, 13)
        s = ShellWangLandau(evaluate, torch.zeros(2, dtype=torch.float64), edges, 0, .4, 1, .25, 0, 129)
        for i in range(60000):
            s.move()
            if (i + 1) % 500 == 0:
                s.refine(.75, 20)
                if s.log_f <= .004:
                    break
        self.assertLessEqual(s.log_f, .004)
        for _ in range(1000):
            s.move(adapt=False)
        for _ in range(40000):
            s.move(adapt=False, collect=True)
        self.assertTrue((s.production > 0).all())
        measured = s.bias + np.log(s.production)
        # Exact bin-average w(y), rather than its midpoint approximation.
        primitive = lambda y: .2 * y + .8 * (2 * y - 1) ** 3 / 6
        expected = np.log(np.diff(primitive(edges)) / np.diff(edges))
        error = measured - expected
        error -= error.mean()
        self.assertLess(np.max(np.abs(error)), .3)
        self.assertGreater(s.round_trips, 2)


if __name__ == "__main__":
    unittest.main()
