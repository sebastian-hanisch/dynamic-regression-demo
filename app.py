"""Dynamische Regression - Prognosen mit bekannten Einflussgrößen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: dieselben Tagesaufträge eines Depots, aber das Modell bekommt den Kalender (Wochentag, Jahr, Feiertage) und den Aktionsplan als Regressoren und
lässt die Fehler von einem ARIMA-Modell erklären. Es behebt die Schwäche der beiden Vorgänger: Feiertage und Aktionen.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import dr_constants as C
import dr_forecast as F
from dr_evaluation import Settings, analyse, diagnostics, effects_experiment, errors_experiment, ladder_experiment, true_effects
from dr_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from dr_visualization import SHORT, build_acf, build_bars, build_effects, build_errors, build_horizon, build_ladder, build_origin, build_series, build_weekly, build_yearly, method_label

st.set_page_config(page_title="Dynamische Regression – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _ladder(seeds):
    return ladder_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _effects(levels, seeds):
    return effects_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _errors(seeds):
    return errors_experiment(seeds=seeds)


st.title("🗓️ Dynamische Regression – Prognosen mit bekanntem Kalender")
st.markdown(
    """
Die Glättung und ARIMA aus den Vorgängern sehen nur die Reihe selbst. Feiertage und Aktionen kennen sie nicht - obwohl der Kalender und der Aktionsplan **vorab bekannt** sind. Die **dynamische Regression** gibt sie dem Modell als **Regressoren**: Wochentag, Trend, Jahresmuster, Feiertage, Aktionen
werden geschätzt wie in einer Regression (jeder Koeffizient ist ein **Effekt**, den man lesen kann), und was übrig bleibt, erklärt ein **ARIMA-Modell für die Fehler**. Die Demo läuft auf **denselben Tagesaufträgen eines Depots** wie die drei Vorgänger und zeigt, wie nah man damit an die
Orakel-Untergrenze kommt, ob die geschätzten Effekte stimmen - und wozu die ARIMA-Fehler gut sind.
"""
)
st.caption(
    "Viertes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe; alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben (die Kreuzprobe im Test läuft gegen statsmodels). "
    "**Bezug zu OR:** Nachfrageprognosen für Bestand, Personal und Touren brauchen die Kalendereffekte - und die Regression liefert sie als Zahlen, die sich mit dem Fachbereich besprechen lassen (\"eine Aktion bringt +22 %\")."
)

with st.expander("So funktioniert die dynamische Regression", expanded=True):
    st.markdown(
        """
1. **Regressoren.** Für jeden Tag stehen bekannte Größen bereit: sechs Wochentags-Indikatoren, ein Trend, sin/cos-Paare für das **Jahresmuster** (Fourier: mit wenigen Paaren lässt sich jede glatte Jahreskurve nachbilden), Indikatoren für **Feiertag** und **Tag nach dem Feiertag** und ein Indikator für **Aktion**.
   Das Modell rechnet im Log: dann wirken alle Effekte als Prozente ($e^\\beta - 1$).
2. **Regression mit ARIMA-Fehlern.** $\\log y_t = x_t'\\beta + \\eta_t$, und $\\eta_t$ folgt einem ARIMA-Modell. Ohne ARMA-Terme ist das die gewöhnliche Kleinste-Quadrate-Regression (OLS); mit ihnen werden $\\beta$ und die Fehlerparameter gemeinsam geschätzt.
3. **Prognose.** $x_{t+j}'\\hat\\beta$ für die kommenden Tage - dafür müssen die Regressoren **auch in der Zukunft bekannt** sein (Kalender: ja; Aktionsplan: nur, wenn er vorab feststeht) - plus die ARIMA-Prognose der Fehler.
4. **Bewertung.** Wie in den Vorgängern: viele Ursprünge (Rolling-Origin), MASE, Orakel-Untergrenze; die Parameter stammen aus den ersten zwei Jahren; SARIMA, Holt-Winters und das Wochenmittel laufen zum Vergleich mit.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario mit Modell laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Regressoren**")
    week = st.checkbox("Wochentag", key="week_check", help="Sechs Indikatoren (Montag ist die Basis).")
    rtrend = st.checkbox("Trend", key="rtrend_check", help="Ein linearer Trend (je Jahr).")
    year_k = st.slider("Jahresmuster (Fourier-Paare)", *bounds("year_slider"), key="year_slider", help="Wie viele sin/cos-Paare das Jahresmuster nachbilden (0 = keines).")
    holiday = st.checkbox("Feiertage", key="holiday_check", help="Indikatoren für den Feiertag und den Tag danach.")
    promo = st.selectbox("Aktionen", list(C.PROMO_MODES), key="promo_select", format_func=lambda m: C.PROMO_LABELS[m], help="Ein Indikator für Aktionstage. 'Zukunft unbekannt': das Modell kennt die Aktionen aus der Vergangenheit, aber im Prognosezeitraum ist keine geplant (Regressor = 0).")
    st.markdown("**Die Fehler**")
    p = st.slider("AR-Ordnung p", *bounds("p_slider"), key="p_slider", help="Wie viele letzte Fehler die Fehlerreihe einrechnet.")
    d = st.slider("Differenzieren d", *bounds("d_slider"), key="d_slider", help="1 = die Fehlerreihe wird einmal differenziert (das Niveau ist eine Irrfahrt; die Konstante entfällt).")
    q = st.slider("MA-Ordnung q", *bounds("q_slider"), key="q_slider", help="Wie viele letzte Ein-Schritt-Fehler die Fehlerreihe einrechnet.")
    logt = st.checkbox("Log-Transformation", key="log_check", help="Das Modell rechnet mit log(Aufträge); Koeffizienten sind dann Prozent-Effekte, die Prognose wird mit exp(f + s²/2) zurückgerechnet.")
    st.markdown("**Die Reihe**")
    trend = st.slider("Trend (% je Jahr)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Lineares Wachstum (oder Schrumpfen) der Aufträge, in Prozent des Ausgangsniveaus je Jahr.")
    weekly = st.slider("Wochenmuster", *bounds("weekly_slider"), key="weekly_slider", step=C.WEEKLY_STEP, help="Stärke des Wochentagsmusters (1 = Standard: Freitag am stärksten, Sonntag am schwächsten; 0 = keines).")
    yearly = st.slider("Jahresmuster", *bounds("yearly_slider"), key="yearly_slider", step=C.YEARLY_STEP, help="Amplitude der jahreszeitlichen Schwankung (Anteil des Niveaus).")
    noise = st.slider("Rauschen", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Streuung des multiplikativen Rauschens (log-normal, im Mittel unverzerrt).")
    shift = st.slider("Niveausprung (%)", *bounds("shift_slider"), key="shift_slider", step=C.SHIFT_STEP, help="Sprung des Niveaus an einem zufälligen Tag im letzten Jahr (0 = keiner), z. B. ein neuer Großkunde. Kein Regressor kennt ihn.")
    events = st.slider("Feiertage und Aktionen", *bounds("events_slider"), key="events_slider", step=C.EVENTS_STEP, help="Stärke der Effekte: am Feiertag ruht das Depot (bis -50 %), am Folgetag Nachholeffekt, dazu drei Aktionswochen je Jahr (bis +50 %).")
    st.markdown("**Der Vergleich**")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird.")
    step = st.slider("Abstand der Ursprünge (Tage)", *bounds("step_slider"), key="step_slider", help="Alle wie viele Tage ein neuer Ursprung beginnt. 1 = jeder Tag des letzten Jahres.")
    k = st.slider("Wochen im Wochenmittel (k)", *bounds("k_slider"), key="k_slider", help="Über wie viele letzte Wochen das Vergleichsverfahren 'Wochenmittel' den Wochentag mittelt.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Rauschen, Aktionstage, Phase des Jahresmusters und den Tag des Niveausprungs fest.")
    st.button("🎲 Neue Reihe generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"trend_slider": int(trend), "weekly_slider": round(float(weekly), 2), "yearly_slider": round(float(yearly), 2), "noise_slider": round(float(noise), 2), "shift_slider": int(shift), "events_slider": round(float(events), 2),
                   "horizon_slider": int(horizon), "step_slider": int(step), "k_slider": int(k), "seed_input": int(seed), "week_check": bool(week), "rtrend_check": bool(rtrend), "year_slider": int(year_k), "holiday_check": bool(holiday),
                   "promo_select": promo, "p_slider": int(p), "d_slider": int(d), "q_slider": int(q), "log_check": bool(logt)})

settings = Settings(int(trend), round(float(weekly), 2), round(float(yearly), 2), round(float(noise), 2), int(shift), round(float(events), 2), int(horizon), int(step), int(k), int(seed), bool(week), bool(rtrend), int(year_k), bool(holiday), promo,
                    int(p), int(d), int(q), bool(logt))
a = analyse(settings)
s = a.series
K = settings.k
model, fl = a.user
spec = model.spec
err_label = spec.error_label

# --- Die Reihe und die Prognose ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Reihe und die Prognose an einem Ursprung")
lo_o, hi_o = int(a.origins[0]), int(a.origins[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor (und der Kalender). Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_series(a, origin), width="stretch", key="series_chart")
st.plotly_chart(build_origin(a, origin), width="stretch", key="origin_chart")
weekday_name = C.WEEKDAYS[(origin - 1) % 7]
st.caption(
    f"Reihe mit {s.n} Tagen (Mittel {de(s.y.mean())} Aufträge je Tag); Ursprung an Tag {origin} (letzter bekannter Tag: ein {weekday_name}). Die dicke Linie ist die Prognose der dynamischen Regression für die nächsten {settings.horizon} Tage, gestrichelt SARIMA, Holt-Winters und das Wochenmittel; "
    "die grüne gepunktete Linie ist der wahre Erwartungswert. Wählen Sie einen Ursprung kurz vor einem Feiertag: nur die Regression kennt ihn."
)

dg = diagnostics(a)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Regressoren", f"{len(model.beta) - (1 if spec.d == 0 else 0)}", help=f"Zahl der Regressoren ohne Konstante: {spec.label}.")
c2.metric("AICc", de(model.aicc, 1), help="Bestraft Parameter; nur zwischen Modellen mit derselben Transformation und denselben Fehlerdifferenzen vergleichbar.")
c3.metric("Streuung der Fehler σ", (f"{de(np.sqrt(model.sigma2), 3)} (Log)" if spec.log else f"{de(np.sqrt(model.sigma2), 1)} Aufträge"), help="Wurzel der mittleren Fehlerquadrate auf den Trainingstagen (ohne die ersten 60 Tage).")
c4.metric("Ljung-Box p (14 Lags)", de(dg["lb_p"], 3), help="Nullhypothese: die Fehler sind unkorreliert. Ein kleiner p-Wert (unter 0,05) zeigt, dass im Fehler noch Struktur steckt.")
arma_txt = []
if spec.p:
    arma_txt.append("φ = " + ", ".join(de(v, 3) for v in model.phi))
if spec.q:
    arma_txt.append("θ = " + ", ".join(de(v, 3) for v in model.theta))
st.caption(f"Modell: {spec.label}; Fehler ARIMA{err_label}" + (f" mit {'; '.join(arma_txt)}" if arma_txt else " (gewöhnliche Kleinste-Quadrate-Regression)") + f". Die Fehler zeigen " +
           ("keine" if dg["lb_p"] >= 0.05 else "noch") + f" Autokorrelation (Ljung-Box, p = {de(dg['lb_p'], 3)}).")

st.markdown("---")

# --- Was das Modell gelernt hat -----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was das Modell gelernt hat")
if not spec.log:
    st.info("Ohne Log-Transformation sind die Koeffizienten Aufträge je Tag, keine Prozente; der Vergleich mit den wahren Effekten des Vehikels (Prozente) entfällt.")
else:
    truth = true_effects(settings)
    rows_e = []
    for name in ("Feiertag", "Tag nach Feiertag", "Aktion"):
        if name in model.names:
            b, se_ = model.coef(name), model.se[model.names.index(name)]
            rows_e.append({"Effekt": name, "geschätzt": f"{'+' if b >= 0 else ''}{de(100 * np.expm1(b), 1)} %", "95-%-Bereich": f"{de(100 * np.expm1(b - 2 * se_), 1)} % bis {de(100 * np.expm1(b + 2 * se_), 1)} %", "wahr im Vehikel": f"{de(100 * np.expm1(truth[name]), 1)} %"})
    if rows_e:
        st.markdown("##### Ereigniseffekte (Prozent des Niveaus)")
        st.dataframe(rows_e, hide_index=True)
        st.caption("Jede Zeile ist ein Koeffizient der Regression, umgerechnet in Prozent ($e^\\beta - 1$); der Bereich ist ±2 Standardfehler. Die letzte Spalte kennt nur das Vehikel: die Effekte, mit denen die Reihe erzeugt wurde.")
    else:
        st.info("Die Ereignisse sind nicht im Modell: es gibt keine Effekte zu lesen - und der Fehler an Feiertagen und Aktionstagen bleibt (siehe die Auswertung unten).")
    g1, g2 = st.columns(2)
    with g1:
        if spec.week:
            st.markdown("##### Wochentagsfaktoren")
            st.plotly_chart(build_weekly(model, settings), width="stretch", key="weekly_chart")
        else:
            st.info("Ohne Wochentag-Regressoren gibt es kein Wochenmuster zu zeigen.")
    with g2:
        if spec.year_k > 0:
            st.markdown("##### Jahresmuster")
            st.plotly_chart(build_yearly(model, settings), width="stretch", key="yearly_chart")
        else:
            st.info("Ohne Fourier-Paare hat das Modell kein Jahresmuster; die Fehlerreihe muss es auffangen.")

with st.expander("Autokorrelation der Fehler"):
    st.plotly_chart(build_acf(dg["acf_res"], f"ACF der Fehler (Trainingstage), ARIMA{err_label}", len(dg["resid"])), width="stretch", key="acf_res_chart")
    st.caption("Wie bei ARIMA: die Balken sind Autokorrelationen, das Band ist ±1,96/√n, die Wochenlags sind orange. Bei vollständigen Regressoren sind die Fehler praktisch weißes Rauschen; fehlt ein Regressor (zum Beispiel das Jahresmuster), bleibt Struktur übrig.")

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Auswertung über alle Ursprünge")
n_org = len(a.origins)
sm = a.summary
u, hw, sa, wm = sm["user"]["mase"], sm["hw_mult"]["mase"], sm["sarima"]["mase"], sm["snaive_k"]["mase"]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Dynamische Regression", f"MASE {de(u, 2)}", delta=f"{u - hw:+.2f}".replace(".", ",") + " gegen Holt-Winters", delta_color="inverse", help="Der Pfeil vergleicht mit Holt-Winters multiplikativ aus Stück 2 (grün: besser).")
m2.metric("Holt-Winters mult.", f"MASE {de(hw, 2)}", help="Das beste Glättungsmodell aus Stück 2.")
m3.metric("SARIMA (Stück 3)", f"MASE {de(sa, 2)}", help="SARIMA(1,0,1)(1,1,1) im Log.")
m4.metric(f"Wochenmittel (k = {K})", f"MASE {de(wm, 2)}", help="Derselbe Wochentag, gemittelt über die letzten k Wochen (Stück 1).")
m5.metric("Orakel-Untergrenze", f"MASE {de(a.oracle['mase'], 2)}", help="Fehler der Prognose 'wahrer Erwartungswert' gegen die beobachteten Werte: das Rauschen der Reihe. Kein Verfahren liegt im Mittel darunter.")
st.plotly_chart(build_bars(a), width="stretch", key="bars_chart")
rows = [{"Verfahren": method_label(m, K), "MASE": de(sm[m]["mase"], 2), "MAE (Aufträge)": de(sm[m]["mae"], 1), "RMSE": de(sm[m]["rmse"], 1), "Verzerrung (Prognose minus Ist)": de(sm[m]["me"], 1)} for m in sorted(sm, key=lambda m: sm[m]["mase"])]
st.dataframe(rows, hide_index=True)
gap_or = u / a.oracle["mase"] - 1
if u < hw - 0.005:
    st.success(f"✅ Die dynamische Regression schlägt Holt-Winters: MASE {de(u, 3)} gegen {de(hw, 3)} (SARIMA {de(sa, 2)}, Wochenmittel {de(wm, 2)}) und liegt {pct(gap_or)} über der Orakel-Untergrenze ({de(a.oracle['mase'], 2)}): die bekannten Einflussgrößen tragen den Vorsprung.")
elif u <= hw + 0.005:
    st.info(f"Dynamische Regression und Holt-Winters liegen gleichauf: MASE {de(u, 3)} gegen {de(hw, 3)} (Orakel {de(a.oracle['mase'], 2)}).")
else:
    st.warning(f"⚠️ Hier verliert die Regression: MASE {de(u, 2)} gegen {de(hw, 2)} bei Holt-Winters (SARIMA {de(sa, 2)}, Orakel {de(a.oracle['mase'], 2)}). Ihr fehlt etwas, das die Glättung von selbst lernt - ein Regressor (Jahresmuster?) oder das Wissen um einen Niveausprung; "
               "probieren Sie ARIMA(0,1,1)-Fehler (d = 1, q = 1).")
st.caption(f"{n_org} Ursprünge im Testjahr (Abstand {settings.step} Tage), je {settings.horizon} Tage Horizont; MASE-Nenner: saisonal naiver Fehler in den ersten {C.FIRST_TEST} Tagen ({de(F.mase_scale(s.y, C.FIRST_TEST), 1)} Aufträge). "
           "Die Regression kennt den Kalender auch für die Zukunft" + (" und den Aktionsplan." if settings.promo == "known" else (" - Aktionen nur aus der Vergangenheit." if settings.promo == "unknown" else ", Aktionen sind nicht im Modell.")))

st.markdown("##### Wie der Fehler mit dem Horizont wächst")
st.plotly_chart(build_horizon(a), width="stretch", key="horizon_chart")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Welcher Regressor bringt was?")
st.caption(f"Standardreihe; Regressoren Schritt für Schritt hinzugefügt (OLS-Fehler, im Log), dazu Aktionen nur aus der Vergangenheit bekannt, ohne Log und die Zahl der Fourier-Paare. Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer wenige Sekunden.")
if st.button("Regressoren durchrechnen", key="ladder_start"):
    st.session_state["ladder_on"] = True
if st.session_state.get("ladder_on"):
    r = _ladder(C.EXP_SEEDS)
    st.plotly_chart(build_ladder(r), width="stretch", key="ladder_chart")
    by = {x["name"]: x["mase"] for x in r["rows"]}
    names = [x["name"] for x in r["rows"]]
    fo = {k: v[0] for k, v in r["fourier"].items()}
    st.warning(
        f"**Befund:** Nur die Konstante: {de(by[names[0]], 2)}; mit dem Wochentag {de(by[names[1]], 2)}, mit dem Trend {de(by[names[2]], 2)} - noch schlechter als das Wochenmittel ({de(r['snaive_k'], 2)}), weil ein festes Niveau der Reihe nicht folgt. Der große Sprung ist das **Jahresmuster** ({de(by[names[3]], 2)}), dann "
        f"die Feiertage ({de(by[names[4]], 2)}) und die Aktionen ({de(by[names[5]], 2)}): das liegt unter Holt-Winters ({de(r['hw_mult'], 2)}) und SARIMA ({de(r['sarima'], 2)}) und nur {pct(by[names[5]] / r['floor'] - 1, 1)} über der Orakel-Untergrenze ({de(r['floor'], 2)}). "
        f"Kennt das Modell die Aktionen nur aus der Vergangenheit, steigt die MASE auf {de(r['unknown'][0], 2)} - immer noch vor Holt-Winters, aber der Aktionsplan ist {de(r['unknown'][0] - by[names[5]], 2)} MASE-Punkte wert. Ohne Log: {de(r['nolog'][0], 2)}. Schon ein Fourier-Paar genügt ({de(fo[1], 3)} gegen {de(fo[2], 3)} bei zwei Paaren; ohne: {de(fo[0], 2)})."
    )

st.markdown("---")

st.subheader("🔬 Stimmen die geschätzten Effekte?")
st.caption(f"Standardreihe mit Ereignisstärke {', '.join(de(x, 2) for x in C.EVENT_LEVELS)}; die Regression schätzt Feiertag, Tag danach und Aktion; verglichen wird mit den Effekten, mit denen das Vehikel die Reihe erzeugt. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Effekte durchrechnen", key="effects_start"):
    st.session_state["effects_on"] = True
if st.session_state.get("effects_on"):
    re_ = _effects(C.EVENT_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_effects(re_), width="stretch", key="effects_chart")
    top = re_[-1]
    hits = [r["coverage"][n] for r in re_ for n in ("Feiertag", "Tag nach Feiertag", "Aktion")]
    st.warning(
        f"**Befund:** Bei Ereignisstärke {de(top['events'], 2)} schätzt das Modell den Feiertag auf {de(top['est']['Feiertag'], 3)} (wahr: {de(top['truth']['Feiertag'], 3)}), den Tag danach auf {de(top['est']['Tag nach Feiertag'], 3)} (wahr: {de(top['truth']['Tag nach Feiertag'], 3)}) und die Aktion auf "
        f"{de(top['est']['Aktion'], 3)} (wahr: {de(top['truth']['Aktion'], 3)}), im Log; die mittleren Standardfehler sind {de(top['se']['Feiertag'], 3)}, {de(top['se']['Tag nach Feiertag'], 3)} und {de(top['se']['Aktion'], 3)}. Über alle Stärken liegt der wahre Wert in {pct(float(np.mean(hits)))} der Reihen und Effekte innerhalb von zwei Standardfehlern. "
        f"Das ist der Unterschied zu den Vorgängern: die Regression sagt nicht nur voraus, sondern **erklärt** - eine Aktion bringt {de(100 * np.expm1(top['est']['Aktion']), 0)} % (wahr: {de(100 * np.expm1(top['truth']['Aktion']), 0)} %). Die Effekte gelten in diesem Vehikel, weil sie multiplikativ wirken und das Modell im Log rechnet."
    )

st.markdown("---")

st.subheader("🔬 Wozu ARIMA-Fehler?")
st.caption(f"Vier Fehlermodelle in vier Szenarien: vollständige Regressoren, ein Niveausprung (+30 %), ein fehlender Jahresregressor und fehlende Ereignis-Regressoren. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Fehlermodelle durchrechnen", key="errors_start"):
    st.session_state["errors_on"] = True
if st.session_state.get("errors_on"):
    rr = _errors(C.EXP_SEEDS)
    st.plotly_chart(build_errors(rr), width="stretch", key="errors_chart")
    std, sh, ny, ne = rr
    names_e = [k for k in std if k not in ("scenario", "n_seeds", "hw_mult", "phi", "lb")]
    ols, ar1, i11, rw = names_e
    st.warning(
        f"**Befund:** Sind die Regressoren vollständig, sind die Fehler weißes Rauschen: AR(1) schätzt φ = {de(std['phi'], 2)} und ändert nichts ({de(std[ols], 3)} gegen {de(std[ar1], 3)}), ARIMA(0,1,1)-Fehler kosten nichts ({de(std[i11], 3)}). Kennt keiner der Regressoren einen **Niveausprung**, versagt die OLS-Regression ({de(sh[ols], 2)}), "
        f"mit ARIMA(0,1,1)-Fehlern schrumpft der Fehler auf {de(sh[i11], 2)} (Holt-Winters {de(sh['hw_mult'], 2)}); die Irrfahrt ohne MA-Term hilft nur wenig ({de(sh[rw], 2)}) und schadet in der Standardreihe ({de(std[rw], 2)}). Fehlt der **Jahresregressor**, rettet AR(1) wenig ({de(ny[ols], 2)} auf {de(ny[ar1], 2)}, φ = {de(ny['phi'], 2)}), ARIMA(0,1,1) aber viel ({de(ny[i11], 2)}, besser als Holt-Winters {de(ny['hw_mult'], 2)}). "
        f"Fehlen die **Ereignis-Regressoren**, sind die ARIMA(0,1,1)-Fehler etwas schlechter als OLS ({de(ne[ols], 2)} gegen {de(ne[i11], 2)}): die Absicherung ist nicht in jedem Szenario gratis. Sie ist eine Versicherung gegen strukturelle Abweichungen, die die Regressoren nicht kennen. **Der Preis:** Sie verschleiert, dass etwas fehlt - der Ljung-Box-Test schlägt bei fehlendem Jahresregressor mit OLS-Fehlern in {pct(ny['lb'][ols])} der Reihen an, mit ARIMA(0,1,1)-Fehlern in {pct(ny['lb'][i11])}."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Regressoren sind auch in der Zukunft bekannt** | Der Kalender ja, der Aktionsplan nur, wenn er vorab feststeht; kennt das Modell die Aktionen nur aus der Vergangenheit, verliert es einen Teil des Vorsprungs (Experiment oben). | Prognose der Regressoren selbst, Szenarien |
| **Alle wichtigen Einflüsse sind Regressoren** | Ein Niveausprung, eine Wetterlage, ein neuer Kunde: was kein Regressor kennt, landet im Fehler. Ohne ARIMA-Fehler versagt die Regression dort; mit ihnen fängt sie es teilweise auf. | Glättung und ARIMA-Fehler im Verbund, Kombination (Stück 9) |
| **Die Effekte sind linear (im Log) und konstant** | Ein Feiertag wirkt hier immer gleich stark; in echten Reihen hängt er vom Wochentag ab oder ändert sich mit der Zeit. Das Modell schätzt einen Mittelwert. | Boosting mit Kalendermerkmalen (Stück 6) |
| **Die Regressoren wurden richtig gewählt** | Zu wenige Fourier-Paare oder ein vergessener Feiertag verzerren die anderen Koeffizienten; das Modell warnt nicht (der Ljung-Box-Test schlägt bei OLS-Fehlern an, ARIMA-Fehler verschleiern das). | Modellwahl, Diagnose |
| **Der Bedarf ist nie null** | Das Log verlangt positive Werte; bei vielen Nullen brechen Log und Regression. | Croston, SBA, TSB (Stück 5) |
| **Es gibt eine Punktprognose** | Die Fehlerstreuung σ liefert Intervalle, aber nur unter der Annahme normalverteilter Fehler. | Prognoseintervalle (Stück 7) |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt - die Regressoren passen exakt zur Erzeugung. In echten Reihen ist der Vorsprung kleiner. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → **Dynamische Regression**, dazu Croston, Boosting, Prognoseintervalle, Hierarchie, Kombination, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $z_t = x_t'\beta + \eta_t$ mit $z_t = \log y_t$ (oder $y_t$) und $\phi(B)\,(1-B)^d\,\eta_t = \theta(B)\,\varepsilon_t$. Regressoren $x_t$: Konstante (nur bei $d = 0$), Wochentags-Indikatoren $\mathbb 1[\mathrm{dow}_t = w]$ ($w = 1..6$), $t/365$, $\sin(2\pi k\,\mathrm{doy}_t/365)$ und $\cos(2\pi k\,\mathrm{doy}_t/365)$ für $k = 1..K$,
Indikatoren für Feiertag, Tag danach und Aktion. Ein Koeffizient $\beta_j$ eines Indikators ist ein Log-Effekt; der Effekt in Prozent ist $e^{\beta_j} - 1$.

**Schätzen (konzentriert).** Für gegebene ARMA-Parameter $\psi = (\phi, \theta)$ sei $\mathcal F_\psi$ die bedingte Fehlerrekursion aus dem ARIMA-Stück. Dann ist $\varepsilon = \mathcal F_\psi(\Delta^d z) - \mathcal F_\psi(\Delta^d X)\,\beta$ linear in $\beta$: für jedes $\psi$ liefern die Normalgleichungen
$\hat\beta(\psi) = (\tilde X'\tilde X)^{-1}\tilde X'\tilde z$ mit den gefilterten Größen $\tilde z, \tilde X$, und die Fehlersumme $\mathrm{SSE}(\psi) = \tilde z'\tilde z - \tilde z'\tilde X\hat\beta$. Nur $\psi$ (höchstens vier Parameter) wird gesucht: 1 200 feste Zufallspunkte, dann sechs Runden lokaler
Verfeinerung, deterministisch; die ersten 60 Tage zählen nicht in die Summe. Standardfehler: $\sqrt{\hat\sigma^2\,\mathrm{diag}((\tilde X'\tilde X)^{-1})}$ mit $\hat\sigma^2 = \mathrm{SSE}/n$. Ohne ARMA-Terme ist das die gewöhnliche Kleinste-Quadrate-Regression.

**Prognose.** Für den Ursprung $t$ und $j = 1..h$: $\hat z_{t+j-1} = x_{t+j-1}'\hat\beta + \hat\eta_{t+j-1}$, wobei $\hat\eta$ die ARIMA-Prognose der Fehlerreihe $\eta_s = z_s - x_s'\hat\beta$, $s < t$, ist (mit ihren eigenen Fehlern aus den Tagen vor $t$); zurückgerechnet mit $\exp(\hat z + \hat\sigma^2/2)$.
Kennt der Aktionsplan die Zukunft nicht, wird der Aktionsregressor in den Prognosetagen auf 0 gesetzt.

Implementiert in `dr_dynreg.py` (Regressoren, Schätzung, Prognose), `dr_sarima.py` (ARIMA-Fehler, aus dem ARIMA-Stück), `dr_ets.py` (Glättung zum Vergleich), `dr_diagnostics.py` (ACF, Ljung-Box), `dr_forecast.py` (Kennzahlen), `dr_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
