"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Regressor- und Fehlerregler, Würfel-Knopf, Permalink-Grenzen, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dr_constants as C
import dr_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 9 and len(at.get("plotly_chart")) == 7 and len(at.success) + len(at.info) + len(at.warning) >= 1
    assert any("schlägt Holt-Winters" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_origin_slider_survives_a_shorter_test_range():
    at = _run(horizon_slider=1, origin_slider=1090)
    _ok(at)
    at.slider(key="horizon_slider").set_value(28).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 28


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Reihe generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_clamped_and_validated():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["trend"] = "999"
    at.query_params["noise"] = "0.31"
    at.query_params["weekly"] = "abc"
    at.query_params["yeark"] = "9"
    at.query_params["promo"] = "teilweise"
    at.query_params["log"] = "0"
    at.query_params["d"] = "5"
    at.query_params["shift"] = "-35"
    at.run()
    _ok(at)
    assert at.session_state["trend_slider"] == C.TREND_MAX and at.session_state["noise_slider"] == 0.3 and at.session_state["weekly_slider"] == C.DEFAULT_WEEKLY
    assert at.session_state["year_slider"] == C.YEAR_K_MAX and at.session_state["promo_select"] == C.DEFAULT_PROMO and at.session_state["log_check"] is False and at.session_state["d_slider"] == C.D_MAX and at.session_state["shift_slider"] == -40


@pytest.mark.parametrize("kw", [dict(week_check=False, rtrend_check=False, year_slider=0, holiday_check=False, promo_select="none"), dict(p_slider=2, d_slider=1, q_slider=2), dict(log_check=False), dict(log_check=False, d_slider=1, q_slider=1),
                                dict(promo_select="unknown"), dict(promo_select="none", holiday_check=False), dict(year_slider=4, p_slider=1), dict(week_check=False, year_slider=0, d_slider=1),
                                dict(trend_slider=C.TREND_MIN, noise_slider=C.NOISE_MAX), dict(trend_slider=C.TREND_MAX, weekly_slider=0.0, yearly_slider=0.0), dict(horizon_slider=C.HORIZON_MAX, step_slider=C.STEP_MAX),
                                dict(shift_slider=-40, k_slider=12), dict(shift_slider=40, k_slider=2, events_slider=1.0), dict(noise_slider=C.NOISE_MIN, events_slider=0.0, horizon_slider=1)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_regressor_switches_change_the_learned_panels():
    at = _run(week_check=False, year_slider=0, holiday_check=False, promo_select="none")
    _ok(at)
    assert len(at.get("plotly_chart")) == 5 and any("keine Effekte zu lesen" in i.value for i in at.info)


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))


def test_ladder_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    next(b for b in at.button if b.key == "ladder_start").click().run()
    _ok(at)
    assert at.session_state["ladder_on"] and any("Nur die Konstante" in w.value for w in at.warning)


def test_effects_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "EVENT_LEVELS", (0.5, 1.0))
    at = _run()
    next(b for b in at.button if b.key == "effects_start").click().run()
    _ok(at)
    assert at.session_state["effects_on"] and any("schätzt das Modell den Feiertag" in w.value for w in at.warning)


def test_errors_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    at = _run()
    next(b for b in at.button if b.key == "errors_start").click().run()
    _ok(at)
    assert at.session_state["errors_on"] and any("Sind die Regressoren vollständig" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
