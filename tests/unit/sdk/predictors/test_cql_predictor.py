# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_checkpoint(tmp_path):
    import torch

    state_dim, action_dim = 40, 8
    hidden_dim, num_blocks = 64, 2

    from aiconfigurator.sdk.predictors.cql_predictor import _build_residual_network

    net = _build_residual_network(state_dim, action_dim, hidden_dim, num_blocks)

    ckpt = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "network_type": "residual",
        "q_net_state_dict": net.state_dict(),
    }
    ckpt_path = tmp_path / "cql_best.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


CALIBRATION_DATA = {
    "meta-llama/Llama-3.1-8B": {
        "ttft_scale": 0.95,
        "ttft_bias": 0.1,
        "tpot_scale": 0.92,
        "tpot_bias": 0.05,
    },
    "_family_LLAMA": {
        "ttft_scale": 0.93,
        "ttft_bias": 0.12,
        "tpot_scale": 0.90,
        "tpot_bias": 0.06,
    },
    "_default": {
        "ttft_scale": 1.0,
        "ttft_bias": 0.0,
        "tpot_scale": 1.0,
        "tpot_bias": 0.0,
    },
}


@pytest.fixture
def mock_checkpoint_with_cal(tmp_path):
    import torch

    state_dim, action_dim = 40, 8
    hidden_dim, num_blocks = 64, 2

    from aiconfigurator.sdk.predictors.cql_predictor import _build_residual_network

    net = _build_residual_network(state_dim, action_dim, hidden_dim, num_blocks)
    ckpt = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "network_type": "residual",
        "q_net_state_dict": net.state_dict(),
    }
    ckpt_path = tmp_path / "cql_best.pt"
    torch.save(ckpt, ckpt_path)

    cal_path = tmp_path / "calibration_params.json"
    with open(cal_path, "w") as f:
        json.dump(CALIBRATION_DATA, f)

    return ckpt_path


class TestCQLPredictorLoadAndPredict:
    def test_from_checkpoint_loads(self, mock_checkpoint):
        from aiconfigurator.sdk.predictors.cql_predictor import CQLPredictor

        predictor = CQLPredictor.from_checkpoint(mock_checkpoint)
        assert predictor._state_dim == 40
        assert predictor._action_dim == 8
        assert predictor._feature_version == 2

    def test_from_checkpoint_with_calibration(self, mock_checkpoint_with_cal):
        from aiconfigurator.sdk.predictors.cql_predictor import CQLPredictor

        predictor = CQLPredictor.from_checkpoint(mock_checkpoint_with_cal)
        assert predictor._calibration is not None
        assert "meta-llama/Llama-3.1-8B" in predictor._calibration

    def test_predict_single_returns_dict(self, mock_checkpoint):
        from aiconfigurator.sdk.predictors.cql_predictor import CQLPredictor

        predictor = CQLPredictor.from_checkpoint(mock_checkpoint)

        with patch(
            "aiconfigurator.sdk.predictors.cql_predictor._encode_state",
            return_value=np.zeros(40, dtype=np.float32),
        ):
            result = predictor.predict_single(
                model_path="meta-llama/Llama-3.1-8B",
                system="h100_sxm", backend="trtllm", version="1.2.0rc5",
                isl=4096, osl=512,
                p_tp=2, p_pp=1, p_bs=4, p_workers=1,
                d_tp=2, d_pp=1, d_bs=64, d_workers=1,
            )

        assert "ttft" in result
        assert "tpot" in result
        assert "feasibility" in result
        assert isinstance(result["ttft"], float)
        assert result["ttft"] >= 0
        assert result["tpot"] >= 0
        assert 0 <= result["feasibility"] <= 1

    def test_predict_batch_returns_dataframe_with_columns_disagg(self, mock_checkpoint):
        from aiconfigurator.sdk.predictors.cql_predictor import CQLPredictor

        predictor = CQLPredictor.from_checkpoint(mock_checkpoint)

        configs = [
            {"p_tp": 2, "p_pp": 1, "p_bs": 4, "p_workers": 1,
             "d_tp": 2, "d_pp": 1, "d_bs": 64, "d_workers": 1},
            {"p_tp": 4, "p_pp": 1, "p_bs": 8, "p_workers": 2,
             "d_tp": 4, "d_pp": 1, "d_bs": 32, "d_workers": 2},
        ]

        with patch(
            "aiconfigurator.sdk.predictors.cql_predictor._encode_state",
            return_value=np.zeros(40, dtype=np.float32),
        ):
            df = predictor.predict_batch(
                model_path="meta-llama/Llama-3.1-8B",
                system="h100_sxm", backend="trtllm", version="1.2.0rc5",
                isl=4096, osl=512, configs=configs,
            )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        from aiconfigurator.sdk.common import ColumnsDisagg
        for col in ColumnsDisagg:
            assert col in df.columns, f"Missing column: {col}"
        assert "feasibility" in df.columns
        assert (df["ttft"] >= 0).all()
        assert (df["tpot"] >= 0).all()


class TestCalibrationLookup:
    def test_model_level_calibration(self, mock_checkpoint_with_cal):
        from aiconfigurator.sdk.predictors.cql_predictor import CQLPredictor

        predictor = CQLPredictor.from_checkpoint(mock_checkpoint_with_cal)
        cal = predictor._get_cal("meta-llama/Llama-3.1-8B")
        assert cal["ttft_scale"] == 0.95

    def test_family_fallback(self, mock_checkpoint_with_cal):
        from aiconfigurator.sdk.predictors.cql_predictor import CQLPredictor

        predictor = CQLPredictor.from_checkpoint(mock_checkpoint_with_cal)
        cal = predictor._get_cal("meta-llama/Llama-3.1-70B")
        assert cal["ttft_scale"] == 0.93

    def test_default_fallback(self, mock_checkpoint_with_cal):
        from aiconfigurator.sdk.predictors.cql_predictor import CQLPredictor

        predictor = CQLPredictor.from_checkpoint(mock_checkpoint_with_cal)
        cal = predictor._get_cal("unknown/SomeModel-7B")
        assert cal["ttft_scale"] == 1.0


class TestRateMatching:
    def test_batch_returns_dataframe(self):
        from aiconfigurator.sdk.predictors.rate_matching import rate_match_batch

        n = 5
        df = rate_match_batch(
            ttft_ms=np.full(n, 10.0),
            tpot_ms=np.full(n, 2.0),
            osl=np.full(n, 512),
            p_tp=np.full(n, 2), p_pp=np.ones(n, dtype=int),
            p_bs=np.full(n, 4), p_workers=np.ones(n, dtype=int),
            d_tp=np.full(n, 2), d_pp=np.ones(n, dtype=int),
            d_bs=np.full(n, 64), d_workers=np.ones(n, dtype=int),
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n
        assert "tokens/s/gpu" in df.columns
        assert (df["tokens/s/gpu"] >= 0).all()

    def test_zero_ttft_handled(self):
        from aiconfigurator.sdk.predictors.rate_matching import rate_match_batch

        df = rate_match_batch(
            ttft_ms=np.array([0.0]),
            tpot_ms=np.array([2.0]),
            osl=np.array([512]),
            p_tp=np.array([2]), p_pp=np.array([1]),
            p_bs=np.array([4]), p_workers=np.array([1]),
            d_tp=np.array([2]), d_pp=np.array([1]),
            d_bs=np.array([64]), d_workers=np.array([1]),
        )
        assert np.isfinite(df["tokens/s/gpu"].values).all()


class TestParetoFront:
    def test_pareto_front_basic(self):
        from aiconfigurator.sdk.predictors.pareto import get_pareto_front

        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
        })
        result = get_pareto_front(df, "x", "y")
        assert len(result) == 5

    def test_pareto_front_dominated(self):
        from aiconfigurator.sdk.predictors.pareto import get_pareto_front

        df = pd.DataFrame({
            "x": [1, 3, 2, 1],
            "y": [4, 3, 3, 1],
        })
        result = get_pareto_front(df, "x", "y")
        assert len(result) == 2
        assert set(result["x"]) == {1, 3}

    def test_filter_feasible(self):
        from aiconfigurator.sdk.predictors.pareto import filter_feasible

        df = pd.DataFrame({
            "feasibility": [0.9, 0.3, 0.7, 0.1],
            "x": [1, 2, 3, 4],
        })
        result = filter_feasible(df)
        assert len(result) == 2
        assert set(result["x"]) == {1, 3}


class TestNetworkBuild:
    def test_build_residual_network(self):
        from aiconfigurator.sdk.predictors.cql_predictor import _build_residual_network

        net = _build_residual_network(40, 8, 64, 2)
        import torch

        x = torch.randn(2, 48)
        out = net(x)
        assert out.shape == (2, 1)

    def test_forward_triple(self):
        from aiconfigurator.sdk.predictors.cql_predictor import _build_residual_network

        net = _build_residual_network(40, 8, 64, 2)
        import torch

        x = torch.randn(3, 48)
        ttft, tpot, feas = net.forward_triple(x)
        assert ttft.shape == (3, 1)
        assert tpot.shape == (3, 1)
        assert feas.shape == (3, 1)
