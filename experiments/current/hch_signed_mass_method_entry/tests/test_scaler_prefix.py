"""The scaler mirror is verified, not trusted.

``CellDataset.numeric_block`` exists so that the harness can fit the core's
``TrainFrozenScaler`` on a fold's training prefix without the core having to
expose an internal accessor.  That makes it a *mirror* of the core's own
canonicalization -- and a mirror that drifts silently would change what every
scaler in every cell is fitted on, which is exactly the kind of quiet
scientific change the entry protocol forbids.

So it is asserted here, bit for bit, against the core's own builder for every
China-5 market and for the synthetic dataset.  If ``src/core`` ever changes the
absolute numeric block, this file fails before any method run does.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.current.hch_signed_mass_method_entry import config as C
from experiments.current.hch_signed_mass_method_entry import synthetic
from experiments.current.hch_signed_mass_method_entry.core_bridge import core

MARKETS = list(C.MARKETS)


def _mirror_matches(dataset, rows) -> None:
    builder = core().DeterministicFeatureBuilder(base_scaler=None)
    tokens = builder.build(dataset.to_raw_inputs(rows)).base_tokens.numpy()
    n_numeric = 1 + dataset.n_features
    numeric, mask = dataset.numeric_block(rows)

    # Exact equality, not ``allclose``: a scaler is fitted on these numbers, so
    # a 1e-7 drift is a real (if tiny) change of the fitted statistic.
    assert np.array_equal(tokens[..., :n_numeric], numeric), dataset.key
    assert numeric.dtype == np.float32, dataset.key

    # The availability mask must agree too, entry for entry.
    available = tokens[..., :n_numeric] != 0.0
    assert np.array_equal(available, (numeric * mask) != 0.0), dataset.key
    # And the mirrored mask itself must equal the core's own view of it.
    assert np.array_equal(numeric != 0.0, (numeric * mask) != 0.0), dataset.key

    # Nothing outside the block may be affected by what the mirror returns.
    assert tokens.shape[-1] == core().base_token_dim(dataset.n_features,
                                                     C.N_CALENDAR_FEATURES)


def test_scaler_prefix_mirror_on_synthetic():
    for n_features in (4, 5, 6, 7, 9):
        dataset = synthetic.synthetic_dataset(n_features=n_features, seed=2)
        _mirror_matches(dataset, np.arange(dataset.n_episodes))


@pytest.mark.parametrize("market", MARKETS)
def test_scaler_prefix_mirror_on_every_china5_market(market):
    from experiments.current.hch_signed_mass_method_entry.runner import load_cell

    checked = 0
    for host in C.HOSTS:
        _contract, _panel, dataset, audit = load_cell(market, host)
        assert audit.protected_final_reads == 0

        # A prefix, both ends of the fitting role, and the whole of it.
        post = dataset.positions(C.ROLE_POST_TRAIN)
        assert post.size > 0
        blocks = [post[:1], post[: max(2, post.size // 2)], post,
                  np.concatenate([dataset.positions(C.ROLE_HOST_TRAIN)[:3], post])]
        for rows in blocks:
            _mirror_matches(dataset, rows)
            checked += 1
    assert checked == 4 * len(C.HOSTS)


def test_scaler_prefix_mirror_leaves_real_missingness_missing():
    """A missing feature must reach the scaler as an explicit zero, not a value."""
    dataset = synthetic.synthetic_dataset(n_features=5, seed=5,
                                          missing_feature_rows=3)
    rows = np.arange(3)
    numeric, mask = dataset.numeric_block(rows)
    assert not mask[:, :, -1].any()
    assert np.all(numeric[:, :, -1] == 0.0)
    _mirror_matches(dataset, rows)


def test_scaler_prefix_fits_only_on_the_rows_it_is_given():
    """The mirror is a pure function of ``rows``; nothing outside can leak in."""
    from experiments.current.hch_signed_mass_method_entry.runner import load_cell

    _contract, _panel, dataset, _audit = load_cell("QINGHAI_DA", "iTransformer")
    post = np.sort(dataset.positions(C.ROLE_POST_TRAIN))
    prefix = post[:8]
    before, _ = dataset.numeric_block(prefix)

    outside = np.setdiff1d(np.arange(dataset.n_episodes), prefix)
    dataset.host_forecast[outside] += 1.0e5
    dataset.features[outside] *= -9.0
    after, _ = dataset.numeric_block(prefix)
    assert np.array_equal(before, after)
