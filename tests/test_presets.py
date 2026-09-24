"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import pytest

import dr_constants as C
import dr_evaluation as E
import dr_presets as P


def _settings(p):
    return E.Settings(p["trend"], p["weekly"], p["yearly"], p["noise"], p["shift"], p["events"], p["horizon"], p["step"], p["k"], p["seed"], p["week"], p["r_trend"], p["year_k"], p["holiday"], p["promo"], p["p"], p["d"], p["q"], p["log"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("trend", "trend_slider"), ("weekly", "weekly_slider"), ("yearly", "yearly_slider"), ("noise", "noise_slider"), ("shift", "shift_slider"), ("events", "events_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9
        assert p["promo"] in C.PROMO_MODES


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: alle Regressoren, OLS-Fehler"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("trend_slider") == (C.TREND_MIN, C.TREND_MAX) and P.bounds("year_slider") == (C.YEAR_K_MIN, C.YEAR_K_MAX) and P.bounds("d_slider") == (0, C.D_MAX)
    assert set(P.STEPS) == {"trend_slider", "weekly_slider", "yearly_slider", "noise_slider", "shift_slider", "events_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_boolean_and_promo_permalink_values_are_validated():
    caster = P.SETTING_SPECS["log_check"].caster
    assert caster("1") is True and caster("0") is False and caster("Ja") is True
    with pytest.raises(ValueError):
        caster("vielleicht")
    promo = P.SETTING_SPECS["promo_select"].caster
    assert promo("known") == "known" and promo("UNKNOWN") == "unknown"
    with pytest.raises(ValueError):
        promo("teilweise")
