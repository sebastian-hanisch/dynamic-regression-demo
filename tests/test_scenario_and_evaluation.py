"""Die Reihe (identisch zu den Vorgängern), die Auswertung, die wahren Effekte des Vehikels und die drei Experimente."""

import numpy as np
import pytest

import dr_constants as C
import dr_dynreg as R
import dr_evaluation as E
import dr_forecast as F
import dr_scenario as S


def test_scenario_is_the_same_vehicle_as_in_the_predecessors():
    ser = S.generate(seed=3)
    assert ser.y[:8].tolist() == [103.0, 131.0, 117.0, 117.0, 127.0, 64.0, 46.0, 132.0] and float(ser.y.sum()) == 126867.0 and float(ser.mu.sum()) == pytest.approx(126496.28723546621)


def test_analyse_is_consistent():
    s = E.Settings(seed=2)
    a = E.analyse(s)
    assert set(a.summary) == set(C.METHODS) and a.best in C.METHODS and len(a.origins) == len(range(C.FIRST_TEST, C.N_DAYS - s.horizon + 1, s.step))
    assert a.oracle["mase"] < min(v["mase"] for v in a.summary.values()) and a.errors["user"].shape == (len(a.origins), s.horizon) and a.X.shape[0] == C.N_DAYS
    y = a.series.y
    loop = np.stack([np.mean([y[t - 7 * (i + 1) + (np.arange(s.horizon) % 7)] for i in range(s.k)], axis=0) for t in a.origins])
    assert a.errors["snaive_k"] + a.actual == pytest.approx(loop)
    assert a.summary["user"]["mase"] == pytest.approx(a.summary["user"]["mae"] / F.mase_scale(y, C.FIRST_TEST))


def test_analyse_with_a_subset_of_methods_gives_the_same_numbers():
    full = E.analyse(E.Settings(seed=2))
    part = E.analyse(E.Settings(seed=2), ("user", "snaive_k"))
    assert set(part.summary) == {"user", "snaive_k"} and part.summary["user"]["mase"] == full.summary["user"]["mase"]


def test_settings_map_to_the_spec_and_the_cache_is_shared():
    s = E.Settings(week=False, r_trend=True, year_k=3, holiday=False, promo="unknown", p=1, d=1, q=2, log=False)
    assert s.spec == R.Spec(False, True, 3, False, True, 1, 1, 2, False) and E.Settings(promo="none").spec.promo is False
    E._fit_dyn.cache_clear()
    E.analyse(E.Settings(seed=9), ("user",))
    n = E._fit_dyn.cache_info().misses
    E.analyse(E.Settings(seed=9, horizon=7, k=8, promo="unknown"), ("user",))
    assert E._fit_dyn.cache_info().misses == n


def test_unknown_promotions_change_only_the_forecast():
    known = E.analyse(E.Settings(seed=2, events=1.0), ("user",))
    unknown = E.analyse(E.Settings(seed=2, events=1.0, promo="unknown"), ("user",))
    assert known.user[0] == unknown.user[0] and unknown.summary["user"]["mase"] > known.summary["user"]["mase"]


def test_horizon_and_step_change_the_result():
    base = E.analyse(E.Settings(seed=2))
    assert len(E.analyse(E.Settings(seed=2, step=7)).origins) == pytest.approx(len(base.origins) / 7, abs=1)
    assert E.analyse(E.Settings(seed=2, horizon=28)).summary["user"]["mae"] != base.summary["user"]["mae"]


def test_true_effects_by_hand():
    t = E.true_effects(E.Settings(events=0.5, weekly=1.0))
    assert t["Sa"] == pytest.approx(np.log(0.55 / 1.10)) and t["So"] == pytest.approx(np.log(0.35 / 1.10)) and t["Feiertag"] == pytest.approx(np.log(0.75)) and t["Tag nach Feiertag"] == pytest.approx(np.log(1.075)) and t["Aktion"] == pytest.approx(np.log(1.25))
    assert E.true_effects(E.Settings(events=0.0))["Feiertag"] == 0.0 and E.true_effects(E.Settings(weekly=0.0))["Fr"] == pytest.approx(0.0)


def test_the_true_yearly_curve_is_the_one_the_generator_uses():
    s = E.Settings(seed=5, yearly=0.3, trend=0, weekly=0.0, events=0.0, shift=0)
    ser = E._series(s.series_key)
    _, model, _ = E._fit_dyn(s.series_key, s.spec)
    doy, est, true = E.yearly_curves(model, s)
    assert true == pytest.approx(ser.mu[:365] / ser.mu[:365].mean(), abs=1e-9) and np.abs(est - true).max() < 0.05


def test_diagnostics_keys_and_shapes():
    a = E.analyse(E.Settings(seed=2), ("user",))
    dg = E.diagnostics(a)
    assert len(dg["acf_res"]) == C.ACF_LAGS and 0 <= dg["lb_p"] <= 1 and len(dg["resid"]) == C.FIRST_TEST - R.BURN
    b = E.analyse(E.Settings(seed=2, p=1, d=1, q=1), ("user",))
    assert len(E.diagnostics(b)["resid"]) == C.FIRST_TEST - 1 - (R.BURN - 1)


def test_experiments_return_consistent_rows():
    lad = E.ladder_experiment(seeds=(0, 1))
    assert [r["name"] for r in lad["rows"]] == [n for n, _ in E.LADDER] and lad["rows"][3]["mase"] < lad["rows"][1]["mase"] and lad["unknown"][0] > lad["rows"][-1]["mase"] and set(lad["fourier"]) == set(C.YEAR_K_LEVELS) and lad["floor"] > 0
    ef = E.effects_experiment(levels=(0.5, 1.0), seeds=(0, 1))
    assert [r["events"] for r in ef] == [0.5, 1.0] and ef[1]["truth"]["Feiertag"] < ef[0]["truth"]["Feiertag"] and ef[1]["est"]["Feiertag"] < ef[0]["est"]["Feiertag"] and all(0 <= v <= 1 for v in ef[0]["coverage"].values())
    er = E.errors_experiment(seeds=(0,))
    assert len(er) == 4 and all(name in er[0] for name, _ in E.ERROR_MODELS) and set(er[0]["lb"]) == {n for n, _ in E.ERROR_MODELS} and er[1]["OLS-Fehler (0,0,0)"] > er[1]["ARIMA(0,1,1)-Fehler"]
