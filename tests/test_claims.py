"""Jede Zahl aus README und PRESET_HELP als Test. Reihen und Schätzung sind deterministisch (fester Seed der Parametersuche); die Bänder sind trotzdem großzügiger als die Rundung, damit andere numpy-/BLAS-Versionen nicht stören."""

import numpy as np
import pytest

import dr_constants as C
import dr_evaluation as E
import dr_presets as P

STD = "Standardfall: alle Regressoren, OLS-Fehler"


def _preset(name, methods=C.METHODS):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["trend"], p["weekly"], p["yearly"], p["noise"], p["shift"], p["events"], p["horizon"], p["step"], p["k"], p["seed"], p["week"], p["r_trend"], p["year_k"], p["holiday"], p["promo"], p["p"], p["d"], p["q"], p["log"]), methods)


def _m(a):
    return {k: v["mase"] for k, v in a.summary.items()}


def _pct(b):
    return 100 * np.expm1(b)


def test_standard_preset():
    a = _preset(STD)
    m = _m(a)
    assert len(a.origins) == 352 and a.oracle["mase"] == pytest.approx(0.747, abs=0.01)
    assert m["user"] == pytest.approx(0.762, abs=0.02) and m["sarima"] == pytest.approx(0.839, abs=0.02) and m["hw_mult"] == pytest.approx(0.841, abs=0.015) and m["snaive_k"] == pytest.approx(0.878, abs=0.006)
    assert m["user"] / a.oracle["mase"] - 1 == pytest.approx(0.02, abs=0.02)
    model = a.user[0]
    assert _pct(model.coef("Feiertag")) == pytest.approx(-27.4, abs=3) and _pct(model.coef("Aktion")) == pytest.approx(24.1, abs=3) and E.diagnostics(a)["lb_p"] == pytest.approx(0.75, abs=0.15)
    truth = E.true_effects(a.settings)
    assert _pct(truth["Feiertag"]) == pytest.approx(-25.0) and _pct(truth["Aktion"]) == pytest.approx(25.0)


def test_error_model_presets():
    a = _preset("Mit ARIMA(0,1,1)-Fehlern")
    assert _m(a)["user"] == pytest.approx(0.756, abs=0.02) and a.user[0].theta[0] == pytest.approx(-0.995, abs=0.01)
    ols, ari = _preset("Niveausprung +30 %, OLS-Fehler"), _preset("Niveausprung +30 %, ARIMA(0,1,1)-Fehler")
    assert ols.series.shift_day == 809 and _m(ols)["user"] == pytest.approx(1.618, abs=0.05) and _m(ols)["hw_mult"] == pytest.approx(1.029, abs=0.02) and _m(ols)["sarima"] == pytest.approx(1.024, abs=0.02) and _m(ols)["snaive_k"] == pytest.approx(1.079, abs=0.006)
    assert _m(ari)["user"] == pytest.approx(1.234, abs=0.05) and _m(ari)["user"] < _m(ols)["user"]
    ny = _preset("Ohne Jahresregressor, ARIMA(0,1,1)-Fehler")
    assert _m(ny)["user"] == pytest.approx(0.807, abs=0.03) and ny.user[0].theta[0] == pytest.approx(-0.892, abs=0.04) and _m(ny)["user"] < _m(ny)["hw_mult"]


def test_no_event_regressors_preset():
    a = _preset("Ohne Ereignis-Regressoren (Stärke 1,0)")
    m = _m(a)
    assert m["user"] == pytest.approx(0.735, abs=0.02) and m["hw_mult"] == pytest.approx(0.808, abs=0.015) and m["sarima"] == pytest.approx(0.793, abs=0.02) and a.oracle["mase"] == pytest.approx(0.616, abs=0.01) and E.diagnostics(a)["lb_p"] < 0.001


def test_yearly_curve_is_recovered():
    a = _preset(STD)
    _, est, true = E.yearly_curves(a.user[0], a.settings)
    assert np.abs(est - true).max() < 0.03


@pytest.fixture(scope="module")
def twelve():
    return [E.analyse(E.Settings(seed=s)) for s in C.EXP_SEEDS]


def test_standard_over_twelve_series(twelve):
    mean = {k: float(np.mean([a.summary[k]["mase"] for a in twelve])) for k in C.METHODS}
    floor = float(np.mean([a.oracle["mase"] for a in twelve]))
    assert mean["user"] == pytest.approx(0.746, abs=0.02) and mean["hw_mult"] == pytest.approx(0.880, abs=0.02) and mean["sarima"] == pytest.approx(0.879, abs=0.02) and mean["snaive_k"] == pytest.approx(0.948, abs=0.02) and floor == pytest.approx(0.735, abs=0.01)
    assert mean["user"] / floor - 1 == pytest.approx(0.015, abs=0.02) and all(a.summary["user"]["mase"] < a.summary["hw_mult"]["mase"] for a in twelve)


def test_ladder_experiment():
    r = E.ladder_experiment()
    mase = [x["mase"] for x in r["rows"]]
    assert mase == pytest.approx([2.28, 1.228, 1.229, 0.819, 0.795, 0.746], abs=0.03) and mase[2] > r["snaive_k"] and mase[5] < r["hw_mult"] - 0.1 and mase[5] / r["floor"] - 1 == pytest.approx(0.015, abs=0.02)
    assert r["unknown"][0] == pytest.approx(0.786, abs=0.03) and r["unknown"][0] < r["hw_mult"] and r["unknown"][0] - mase[5] == pytest.approx(0.04, abs=0.03) and r["nolog"][0] == pytest.approx(0.834, abs=0.03)
    fo = {k: v[0] for k, v in r["fourier"].items()}
    assert fo[0] == pytest.approx(1.20, abs=0.05) and [round(fo[k], 3) for k in (1, 2, 3, 4)] == pytest.approx([0.748, 0.746, 0.747, 0.749], abs=0.02) and r["hw_mult"] == pytest.approx(0.880, abs=0.01) and r["sarima"] == pytest.approx(0.879, abs=0.01)


def test_effects_experiment():
    rows = {r["events"]: r for r in E.effects_experiment()}
    top = rows[1.0]
    assert top["est"]["Feiertag"] == pytest.approx(-0.688, abs=0.03) and top["truth"]["Feiertag"] == pytest.approx(-0.693, abs=1e-3) and top["est"]["Tag nach Feiertag"] == pytest.approx(0.151, abs=0.04) and top["truth"]["Tag nach Feiertag"] == pytest.approx(0.140, abs=1e-3)
    assert top["est"]["Aktion"] == pytest.approx(0.407, abs=0.02) and top["truth"]["Aktion"] == pytest.approx(0.405, abs=1e-3) and top["se"]["Feiertag"] == pytest.approx(0.033, abs=0.005) and top["se"]["Aktion"] == pytest.approx(0.024, abs=0.004)
    assert top["est"]["Sa"] == pytest.approx(-0.688, abs=0.03) and top["truth"]["Sa"] == pytest.approx(-0.693, abs=1e-3)
    assert all(rows[ev]["coverage"][n] >= 0.9 for ev in rows for n in ("Feiertag", "Tag nach Feiertag", "Aktion"))
    assert all(abs(rows[ev]["est"]["Feiertag"] - rows[ev]["truth"]["Feiertag"]) < 0.03 for ev in rows)


def test_errors_experiment():
    std, sh, ny, ne = E.errors_experiment()
    ols, ar1, i11, rw = (name for name, _ in E.ERROR_MODELS)
    assert std[ols] == pytest.approx(0.746, abs=0.02) and std[ar1] == pytest.approx(0.746, abs=0.02) and std[i11] == pytest.approx(0.743, abs=0.02) and std[rw] == pytest.approx(1.065, abs=0.05) and std["phi"] == pytest.approx(-0.01, abs=0.05) and std["hw_mult"] == pytest.approx(0.880, abs=0.01)
    assert sh[ols] == pytest.approx(1.407, abs=0.05) and sh[ar1] == pytest.approx(1.408, abs=0.05) and sh[i11] == pytest.approx(1.091, abs=0.04) and sh[rw] == pytest.approx(1.261, abs=0.05) and sh["hw_mult"] == pytest.approx(1.059, abs=0.02)
    assert ny[ols] == pytest.approx(1.200, abs=0.05) and ny[ar1] == pytest.approx(1.152, abs=0.05) and ny[i11] == pytest.approx(0.789, abs=0.03) and ny["phi"] == pytest.approx(0.484, abs=0.06) and ny[i11] < ny["hw_mult"]
    assert ne[ols] == pytest.approx(0.805, abs=0.03) and ne[i11] == pytest.approx(0.838, abs=0.03) and ne[ols] < ne[i11] < ne["hw_mult"] == pytest.approx(0.895, abs=0.02)
    assert ny["lb"][ols] >= 0.9 and ny["lb"][i11] <= 0.1 and std["lb"][ols] <= 0.1 and ne["lb"][ols] >= 0.9


def test_ljung_box_sees_a_missing_regressor_only_with_ols_errors():
    def share(**kw):
        return float(np.mean([E.diagnostics(E.analyse(E.Settings(seed=s, **kw), ("user",)))["lb_p"] < 0.05 for s in C.EXP_SEEDS]))
    assert share(events=1.0) == 0.0 and share(events=1.0, holiday=False, promo="none") >= 0.9 and share(events=0.0, holiday=False, promo="none") <= 0.1 and share(year_k=0) >= 0.9 and share(year_k=0, d=1, q=1) <= 0.1
