# 🗓️ Dynamische Regression – Prognosen mit bekanntem Kalender

Viertes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Nachfolger von [Naiver Prognose](https://github.com/sebastian-hanisch/naive-forecast-demo), [Exponentieller Glättung](https://github.com/sebastian-hanisch/exponential-smoothing-demo) und [ARIMA](https://github.com/sebastian-hanisch/arima-demo);
er behebt die Schwäche, die alle drei teilen: **Feiertage und Aktionen**. Geplant sind sieben weitere Stücke (Croston-Verfahren, Boosting, Prognoseintervalle, Hierarchische Abstimmung, Kombination, Prognose → Bestand, ein vortrainiertes Netz; noch nicht gebaut).

Die Vorgänger sehen nur die Reihe selbst – obwohl der Kalender und der Aktionsplan **vorab bekannt** sind. Die **dynamische Regression** gibt sie dem Modell als **Regressoren**: Wochentag, Trend, Jahresmuster (Fourier-Paare), Feiertag, Tag nach dem Feiertag, Aktion. Die Koeffizienten werden wie in einer Regression geschätzt (jeder ist ein **Effekt**, den man lesen kann),
und was übrig bleibt, erklärt ein **ARIMA-Modell für die Fehler**. Die Demo läuft auf **denselben Tagesaufträgen eines Depots** wie die Vorgänger (dieselbe Reihe, im Test auf denselben Fingerabdruck geprüft), im selben Rolling-Origin-Vergleich mit MASE und Orakel-Untergrenze. Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben; die Kreuzprobe im Test läuft gegen statsmodels.

**Bezug zu OR:** Nachfrageprognosen für Bestand, Personal und Touren brauchen die Kalendereffekte – und die Regression liefert sie als Zahlen, die sich mit dem Fachbereich besprechen lassen ("eine Aktion bringt +24 %").

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie sah für dieses Stück vor: die Regression behebt die Ereignis-Schwäche von Glättung und ARIMA. Die Messung bestätigt das – und zeigt, dass mehr dahintersteckt als die Ereignisse:

1. **Fast die Orakel-Untergrenze.** Mit allen Regressoren erreicht die Regression im Mittel über zwölf Reihen MASE **0,746** gegen 0,880 bei Holt-Winters und 0,879 bei SARIMA – nur 1,5 % über der Untergrenze (0,735).
2. **Der große Sprung ist das Jahresmuster, nicht der Feiertag.** Das Jahresmuster (365 Tage) konnten Glättung und ARIMA nicht abbilden; mit zwei Fourier-Paaren fällt die MASE von 1,23 auf 0,82, die Feiertage bringen dann 0,80 und die Aktionen 0,75. Schon ein Paar genügt (0,748).
3. **Die Effekte stimmen.** Der Feiertag wird auf −0,688 im Log geschätzt (wahr −0,693), die Aktion auf +0,407 (wahr +0,405), der wahre Wert liegt in mindestens 90 % der Reihen (gemessen: allen) innerhalb von zwei Standardfehlern. Das ist der Unterschied zu allen Vorgängern: die Regression sagt nicht nur voraus, sie **erklärt**.
4. **Regressoren allein reichen nicht – dann helfen ARIMA-Fehler.** Bei einem Niveausprung (+30 %) versagt die Regression mit OLS-Fehlern (1,41 gegen 1,06 bei Holt-Winters); mit ARIMA(0,1,1)-Fehlern schrumpft sie auf 1,09, bei vollständigen Regressoren kosten die ARIMA-Fehler nichts (0,743 gegen 0,746). Fehlt der Jahresregressor, rettet ARIMA(0,1,1) die Regression von 1,20 auf 0,79.
5. **Die Absicherung verschleiert.** Der Ljung-Box-Test schlägt bei OLS-Fehlern an, wenn ein Regressor fehlt (ohne Jahresregressor in mindestens 90 % der Reihen, gemessen: 12 von 12), mit ARIMA(0,1,1)-Fehlern in höchstens 10 % (gemessen: nie) – obwohl der Regressor fehlt. Im ARIMA-Stück sah der Test die Ereignisse nicht; hier sieht er sie bei OLS-Fehlern.
6. **Die Zukunft muss bekannt sein.** Kennt das Modell die Aktionen nur aus der Vergangenheit (im Prognosezeitraum keine geplant), steigt die MASE von 0,746 auf 0,786 – immer noch vor Holt-Winters, aber der Aktionsplan ist 0,04 MASE-Punkte wert.

## Modell

- **Die Reihe** (`dr_scenario.py`): wortgleich zu den Vorgängern: 1 095 Tage, Niveau, Trend, Wochenmuster, Jahresmuster, Niveausprung, Feiertage, Aktionen, multiplikatives Rauschen; Ursprünge im letzten Jahr (ab Tag 730).
- **Regressoren** (`dr_dynreg.py`): Konstante (nur bei $d = 0$), sechs Wochentags-Indikatoren, Trend $t/365$, $\sin/\cos(2\pi k\,\mathrm{doy}/365)$ für $k = 1..K$ ($K \le 4$), Feiertag, Tag nach dem Feiertag, Aktion. Das Modell rechnet in $\log y$: alle Effekte wirken multiplikativ, ein Koeffizient ist ein Log-Effekt, $e^\beta - 1$ der Effekt in Prozent.
- **Regression mit ARIMA-Fehlern:** $\log y_t = x_t'\beta + \eta_t$, $\phi(B)(1-B)^d \eta_t = \theta(B)\,\varepsilon_t$ mit $p, q \le 2$ und $d \le 1$; ohne ARMA-Terme ist es die gewöhnliche Kleinste-Quadrate-Regression (OLS).
- **Schätzen (konzentriert):** für gegebene ARMA-Parameter ist das Problem linear in $\beta$: die Reihe und jede Regressorspalte laufen durch dieselbe bedingte Fehlerrekursion (alle Kandidaten und Spalten gleichzeitig), $\beta$ folgt aus den Normalgleichungen der gefilterten Größen; gesucht werden nur die ARMA-Parameter (1 200 feste Zufallspunkte, sechs Runden lokaler Verfeinerung, deterministisch; die ersten 60 Tage zählen nicht).
  Standardfehler aus den gefilterten Normalgleichungen.
- **Prognose:** $x_{t+j}'\hat\beta$ mit den für die Zukunft bekannten Regressoren (Kalender, auf Wunsch der Aktionsplan) plus die ARIMA-Prognose der Fehlerreihe; Rücktransformation $\exp(\hat z + \hat\sigma^2/2)$.
- **Vergleich:** SARIMA(1,0,1)(1,1,1) im Log (Stück 3, `dr_sarima.py`), Holt-Winters multiplikativ (Stück 2, `dr_ets.py`), das Wochenmittel (Stück 1), die Orakel-Untergrenze; ACF und Ljung-Box der Fehler (`dr_diagnostics.py`).

## Methodik

- **Handrechnungen:** die Regressoren spaltenweise (Wochentage, Periode 365 der Fourier-Terme, Feiertag, Tag danach, Aktion), Regressionsprognose (Do/Fr/Sa: Konstante plus Wochentagskoeffizient), AR(1)-Fehleranteil ($0{,}5^j\,\eta_{t-1}$), differenzierte Fehler halten das letzte Niveau, Log mit Bias-Korrektur, Aktionen in der Zukunft auf 0.
- **Kreuzprobe gegen statsmodels:** die Prognose für vier Fehlermodelle – ARMA(1,1), reine OLS, ARIMA(1,1,1) und (0,1,1) – stimmt bei festen Parametern mit `ARIMA(..., exog=...)` auf 1e-5 überein (Regression mit ARIMA-Fehlern in der Form von statsmodels). Bei OLS stimmen Koeffizienten (numpy) und Standardfehler (`sm.OLS`) überein.
- **Konzentrierte Schätzung = gefilterte Fehlerreihe:** die Fehlersumme des Optimums ist die Fehlersumme der Fehlerreihe $z - X\hat\beta$ durch dasselbe Fehlermodell (auf 1e-6). Ein simulierter AR(1)-Fehlerprozess wird wiedererkannt ($\varphi$, $\beta$, $\sigma^2$).
- **Wahre Effekte:** das Vehikel kennt seine Effekte (Feiertag $\log(1 - 0{,}5\,s)$, Tag danach $\log(1 + 0{,}15\,s)$, Aktion $\log(1 + 0{,}5\,s)$, Wochentage aus dem Muster, das Jahresmuster mit der Phase des Seeds); die Tests vergleichen die Schätzung damit.
- **Kein Blick in die Zukunft:** ändert man die Tage ab einem Ursprung, bleibt die Prognose früherer Ursprünge unverändert.
- **Statistik:** zwölf feste Seeds, Fehlerbalken = Standardfehler.
- **Literatur** (nicht nachgebaut): Hyndman/Athanasopoulos, *Forecasting: Principles and Practice* (3. Aufl., Kap. 10, Dynamische Regression; Fourier-Terme Kap. 12); Pankratz, *Forecasting with Dynamic Regression Models* (1991).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardreihen** (12 Seeds, Horizont 14, 352 Ursprünge) | MASE: dynamische Regression (alle Regressoren, OLS-Fehler) **0,746**, Holt-Winters multiplikativ 0,880, SARIMA 0,879, Wochenmittel (k = 4) 0,948; Orakel **0,735** (Regression 1,5 % darüber); vor Holt-Winters in allen zwölf Reihen. | `test_standard_over_twelve_series` |
| Standardfall (Preset, Seed 3) | Regression 0,76, Holt-Winters 0,84, SARIMA 0,84, Wochenmittel 0,88, Orakel 0,75 (2 % darüber). Feiertag −27 % (wahr −25 %), Aktion +24 % (wahr +25 %); Ljung-Box p = 0,75. | `test_standard_preset` |
| **Welcher Regressor bringt was?** (OLS-Fehler, 12 Seeds) | Nur Konstante **2,28**; + Wochentag **1,23**; + Trend 1,23 (schlechter als das Wochenmittel: ein festes Niveau folgt der Reihe nicht); + Jahr (2 Paare) **0,82**; + Feiertage 0,80; + Aktionen **0,75**. Aktionen nur aus der Vergangenheit bekannt: 0,79; ohne Log: 0,83. Fourier-Paare 0 / 1 / 2 / 3 / 4: 1,20 / 0,748 / 0,746 / 0,747 / 0,749. | `test_ladder_experiment` |
| **Stimmen die Effekte?** (Ereignisstärke 0,25 bis 1,0) | Stärke 1,0, Log-Effekte: Feiertag −0,688 (wahr −0,693, Standardfehler 0,033), Tag danach 0,151 (wahr 0,140), Aktion 0,407 (wahr 0,405, Standardfehler 0,024), Samstag −0,688 (wahr −0,693). Der wahre Wert liegt bei Feiertag, Tag danach und Aktion über alle Stärken in mindestens 90 % (gemessen: 100 %) der Reihen innerhalb von zwei Standardfehlern. | `test_effects_experiment` |
| Das Jahresmuster | Die geschätzte Jahreskurve (zwei Fourier-Paare) weicht von der wahren um höchstens 0,03 (gemessen: 0,01) ab. | `test_yearly_curve_is_recovered` |
| **Wozu ARIMA-Fehler?** (MASE, 12 Seeds; OLS / AR(1) / ARIMA(0,1,1) / Irrfahrt (0,1,0); Holt-Winters) | *Standardreihe:* 0,746 / 0,746 / **0,743** / 1,065; φ des AR(1) −0,01; Holt-Winters 0,880. *Niveausprung +30 %:* 1,407 / 1,408 / **1,091** / 1,261; Holt-Winters 1,059. *Jahresregressor fehlt:* 1,200 / 1,152 / **0,789** / 1,069 (φ = 0,48); Holt-Winters 0,880. *Ereignis-Regressoren fehlen (Stärke 1,0):* **0,805** / 0,805 / 0,838 / 1,158; Holt-Winters 0,895. | `test_errors_experiment` |
| Ljung-Box-Test (Niveau 5 %) | Bei vollständigen Regressoren lehnt er in 0 von 12 Reihen ab, ohne Ereignis-Regressoren (Stärke 1,0) und ohne Jahresregressor in mindestens 90 % (gemessen: 12 von 12); mit ARIMA(0,1,1)-Fehlern ohne Jahresregressor in höchstens 10 % (gemessen: 0 von 12): die Fehlerreihe verschleiert den fehlenden Regressor. | `test_ljung_box_sees_a_missing_regressor_only_with_ols_errors`, `test_errors_experiment` |
| Presets: Fehlermodelle (Seed 3) | ARIMA(0,1,1)-Fehler: 0,76 (θ = −0,995). Niveausprung +30 % (Tag 809): OLS 1,62, ARIMA(0,1,1) 1,23; Holt-Winters 1,03, SARIMA 1,02, Wochenmittel 1,08, Orakel 0,90. Ohne Jahresregressor mit ARIMA(0,1,1): 0,81 (θ = −0,89) gegen Holt-Winters 0,84. | `test_error_model_presets` |
| Preset ohne Ereignis-Regressoren (Seed 3, Stärke 1,0) | Regression 0,74, SARIMA 0,79, Holt-Winters 0,81, Orakel 0,62; Ljung-Box p < 0,001. Im Mittel über 12 Reihen 0,81 gegen 0,90 bei Holt-Winters. | `test_no_event_regressors_preset`, `test_errors_experiment` |

Die Preset-Zeilen sind **Einzelreihen** (Seed 3); belastbar sind die Zeilen über zwölf Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Regressoren sind auch in der Zukunft bekannt** | Der Kalender ja, der Aktionsplan nur, wenn er vorab feststeht; kennt das Modell die Aktionen nur aus der Vergangenheit, verliert es einen Teil des Vorsprungs (0,746 → 0,786). | Prognose der Regressoren selbst, Szenarien |
| **Alle wichtigen Einflüsse sind Regressoren** | Ein Niveausprung, eine Wetterlage, ein neuer Kunde: was kein Regressor kennt, landet im Fehler. Mit OLS-Fehlern versagt die Regression dort (1,41), ARIMA(0,1,1)-Fehler fangen es teilweise auf (1,09) – und verschleiern, dass etwas fehlt. | Glättung und ARIMA-Fehler im Verbund, Kombination (geplant) |
| **Die Effekte sind linear (im Log) und konstant** | Ein Feiertag wirkt hier immer gleich stark und multiplikativ; in echten Reihen hängt er vom Wochentag ab oder ändert sich mit der Zeit. Das Modell schätzt einen Mittelwert. | Boosting mit Kalendermerkmalen (geplant) |
| **Die Regressoren wurden richtig gewählt** | Zu wenige Fourier-Paare oder ein vergessener Feiertag verzerren die anderen Koeffizienten; der Ljung-Box-Test schlägt bei OLS-Fehlern an, ARIMA-Fehler verschleiern das. | Modellwahl, Diagnose |
| **Der Bedarf ist nie null** | Das Log verlangt positive Werte; bei vielen Nullen brechen Log und Regression. | Croston, SBA, TSB (geplant) |
| **Es gibt eine Punktprognose** | Die Fehlerstreuung σ liefert Intervalle, aber nur unter der Annahme normalverteilter Fehler. | Prognoseintervalle (geplant) |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt – die Regressoren passen exakt zur Erzeugung (multiplikative Effekte, ein sinusförmiges Jahresmuster). In echten Reihen ist der Vorsprung kleiner. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund anderthalb Minuten, 109 Tests): Regressoren spaltenweise, gefilterte Kleinste Quadrate gegen numpy und statsmodels (Koeffizienten, Standardfehler), konzentrierte Schätzung gegen die gefilterte Fehlerreihe, Wiedererkennung eines simulierten AR-Fehlerprozesses, Prognose von Hand (Regression, ARMA-Fehler, differenzierte Fehler, Log, unbekannte Aktionen),
Kreuzprobe der Prognose gegen `ARIMA(exog=...)` (vier Fehlermodelle), der ARIMA-Kern aus Stück 3 (Polynome, Differenzen, Sonderfälle, Kreuzprobe), ACF/PACF/Ljung-Box gegen statsmodels, die Reihe (Fingerabdruck wie in den Vorgängern), die wahren Effekte des Vehikels, Auswertung und Experimentzeilen, Preset- und Permalink-Klemmen, AppTest-Rauchtests (Standard, jedes Preset, Regressor- und Fehlerregler, Extremwerte,
drei Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen; Reihen und Schätzung sind deterministisch, die Bänder großzügiger als die Rundung).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `dr_constants.py` | Regler-Grenzen, Modelle, Experiment-Seeds |
| `dr_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `dr_scenario.py` | Die Reihe (wortgleich zu den Vorgängern) |
| `dr_dynreg.py` | Regressoren, konzentrierte Schätzung, Prognose |
| `dr_sarima.py` | ARIMA-Kern aus Stück 3 (Fehlermodell, SARIMA-Vergleich) |
| `dr_ets.py` | Glättung aus Stück 2 (Vergleich) |
| `dr_diagnostics.py` | ACF, PACF, Ljung-Box, Chi-Quadrat |
| `dr_forecast.py` | Rolling-Origin-Kennzahlen, Wochenmittel |
| `dr_evaluation.py` | Analyse, wahre Effekte, Diagnose, drei Experimente |
| `dr_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Exakte Likelihood; Regressoren mit Verzögerungen (Übertragungsfunktion); nichtlineare Effekte; Wechselwirkungen (Feiertag × Wochentag).
- Automatische Regressorauswahl (Lasso, AICc über viele Teilmengen); Prognose der Regressoren selbst.
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Kreuzproben im Test: statsmodels, scipy).
