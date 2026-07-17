import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from psm_final.analysis.vdvae_analysis import VDVAEAnalysis
from psm_final.models import vdvae

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT = vdvae.checkpoint_path(_REPO_ROOT)
_needs_checkpoint = unittest.skipUnless(
    _CHECKPOINT.exists(),
    f"VDVAE checkpoint not downloaded ({_CHECKPOINT}); run `python -m psm_final.models.vdvae`",
)


def _stimulus(size=(120, 90), seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


class LatentHierarchyTests(unittest.TestCase):
    """The latent geometry is derived from `dec_blocks`, so these guard the parsing
    that every embedding size depends on."""

    def test_imagenet64_has_75_latent_groups_coarse_to_fine(self):
        resolutions = vdvae.latent_resolutions()

        self.assertEqual(len(resolutions), 75)
        # The resolution cut in `n_top_groups` is only a depth cut because the stack
        # is monotonically coarse -> fine. If upstream ever reordered it, that
        # assumption -- not just this test -- would break.
        self.assertEqual(resolutions, sorted(resolutions))
        self.assertEqual((resolutions[0], resolutions[-1]), (1, 64))

    def test_matches_brain_diffuser_published_dimensionality(self):
        # Ozcelik & VanRullen (arXiv:2303.05334) take the first 31 latent groups of
        # this same checkpoint and report 91,168 dims. Reproducing that number exactly
        # cross-checks our block parsing, zdim and group ordering against a
        # published, independent use of the same model.
        sizes = vdvae.latent_group_sizes()

        self.assertEqual(sum(sizes[:31]), 91_168)

    def test_top_resolution_cuts_have_expected_sizes(self):
        # The cut is parameterised even though only res<=4 runs; pin the sizes across
        # the range. res<=16 == 30 groups is Brain-Diffuser's low-level branch.
        cuts = [(1, 2, 32), (4, 6, 1_056), (8, 14, 9_248), (16, 30, 74_784)]
        for max_resolution, groups, dims in cuts:
            with self.subTest(max_resolution=max_resolution):
                self.assertEqual(vdvae.n_top_groups(max_resolution), groups)
                self.assertEqual(vdvae.embedding_dim(max_resolution), dims)

    def test_default_cut_is_comparable_to_the_beta_vae_latent(self):
        # The arm only replaces the beta-VAE fairly if the embedding stays the same
        # order of magnitude as its 512-d latent.
        self.assertEqual(vdvae.embedding_dim(vdvae.TOP_LATENT_RESOLUTION), 1_056)


class PreprocessTests(unittest.TestCase):
    def test_returns_nhwc_not_nchw(self):
        # VDVAE's encoder permutes (0, 3, 1, 2) itself, so it wants NHWC. Feeding it
        # NCHW would silently "work" on a square image while treating channels as
        # spatial rows -- hence an explicit test.
        batch = vdvae.preprocess(_stimulus())

        self.assertEqual(batch.shape, (1, 64, 64, 3))

    def test_applies_imagenet64_normalisation_not_unit_scaling(self):
        white = Image.new("RGB", (80, 80), (255, 255, 255))
        black = Image.new("RGB", (80, 80), (0, 0, 0))

        self.assertAlmostEqual(
            float(vdvae.preprocess(white).max()), (255 + vdvae.SHIFT) * vdvae.SCALE, places=4)
        self.assertAlmostEqual(
            float(vdvae.preprocess(black).min()), (0 + vdvae.SHIFT) * vdvae.SCALE, places=4)

    def test_converts_grayscale_stimuli_to_rgb(self):
        # Triple-N stimuli are .bmp and may be single-channel.
        batch = vdvae.preprocess(Image.new("L", (80, 80), 128))

        self.assertEqual(batch.shape[-1], 3)


class DiscoverTests(unittest.TestCase):
    def test_skips_arm_when_checkpoint_absent_rather_than_downloading(self, ):
        specs = VDVAEAnalysis.discover(triple_n_path="/nonexistent",
                                       checkpoints_root="/nonexistent")

        self.assertEqual(specs, [])

    @_needs_checkpoint
    def test_offers_one_labelled_model_when_checkpoint_present(self):
        specs = VDVAEAnalysis.discover(triple_n_path="/nonexistent",
                                       checkpoints_root=str(_REPO_ROOT))

        # A single arm at the default cut (the res<=4/8/16 layer sweep is off).
        self.assertEqual(len(specs), 1)
        self.assertIn("VDVAE", specs[0][0])
        self.assertIn(f"res≤{vdvae.TOP_LATENT_RESOLUTION}", specs[0][0])


@_needs_checkpoint
class PretrainedCheckpointTests(unittest.TestCase):
    """End-to-end against the real 500 MB released checkpoint."""

    @classmethod
    def setUpClass(cls):
        # CPU keeps this deterministic and runnable without a GPU.
        cls.model = vdvae.load_vdvae(_CHECKPOINT, device="cpu")

    def test_declared_hyperparameters_match_the_released_weights(self):
        # load_vdvae uses strict=True, so setUpClass succeeding IS the assertion that
        # IMAGENET64_HPS is right. This pins the param count as a second signal.
        n_params = sum(p.numel() for p in self.model.parameters())

        self.assertAlmostEqual(n_params / 1e6, 125, delta=10)

    def test_embedding_has_the_advertised_width(self):
        batch = vdvae.preprocess(_stimulus(), device="cpu")

        latents = vdvae.top_latents(self.model, batch)

        self.assertEqual(latents.shape, (1, vdvae.embedding_dim()))

    def test_embedding_is_deterministic(self):
        # The whole point of the vendored patches. Without them every RDM built from
        # this arm would be irreproducible run to run.
        batch = vdvae.preprocess(_stimulus(), device="cpu")

        first = vdvae.top_latents(self.model, batch)
        second = vdvae.top_latents(self.model, batch)

        torch.testing.assert_close(first, second)

    def test_sampled_pass_is_stochastic_beyond_the_first_group(self):
        # Characterises *why* `deterministic=True` is required rather than just
        # reading qm off a normal pass: VDVAE is autoregressive over its hierarchy,
        # so upstream sampling leaks into every deeper group's posterior mean. Group 0
        # is conditioned only on the learned bias, so it stays put; group 5 does not.
        batch = vdvae.preprocess(_stimulus(), device="cpu")

        first = self.model.forward_get_latents(batch, deterministic=False)
        second = self.model.forward_get_latents(batch, deterministic=False)

        torch.testing.assert_close(first[0]["qm"], second[0]["qm"])
        self.assertGreater(float((first[5]["qm"] - second[5]["qm"]).abs().max()), 1e-3)

    def test_distinguishes_different_stimuli(self):
        # A guard against the embedding collapsing to a constant (e.g. if the cut
        # selected only prior-driven groups), which would make every RDM entry equal.
        one = vdvae.top_latents(self.model, vdvae.preprocess(_stimulus(seed=1), device="cpu"))
        two = vdvae.top_latents(self.model, vdvae.preprocess(_stimulus(seed=2), device="cpu"))

        self.assertGreater(float((one - two).abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
