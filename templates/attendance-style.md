# LaTeX-Stil der Anwesenheitsaufgaben

Referenz ist der vom Autor überarbeitete Part I in
`attendance_exercises/1_automatic_differentiation.md` (25. September 2026).
Diese Regeln ergänzen die technische Pandoc-Vorlage `course-sheet.latex`.
Die redaktionelle Auswahl und Gruppierung von Gleichungen erfolgt bei der
Überarbeitung des Markdown-Texts, nicht durch bloßes Kompilieren.

## Änderungen gegenüber der ersten Konvertierung

- Flachere Hierarchie: Theorie als `Part I - …`, `Part II - …` usw., darunter
  direkt nummerierte Unterabschnitte. Zusätzliche Sammelüberschriften wie
  „Dual number basics“ entfallen.
- Der jeweilige Aufgabenblock ist eine eigene Section. Einzelne Aufgaben
  sind Subsections mit expliziten Bezeichnungen wie „Exercise 1.1 - …“.
- Zentrale Definitionen und Regeln erhalten Gleichungsnummern. Beispiele
  und Hilfsrechnungen bleiben überwiegend unnummeriert.
- Zusammengehörige Formeln stehen in einer gemeinsamen Umgebung; mehrzeilige
  Rechnungen werden mit `split` ausgerichtet.
- Kurze Angaben wie Auswertungspunkte und Rechenziele stehen im Fließtext.
  Die erklärenden Sätze verbinden die Rechenschritte unmittelbar.
- Rechenregeln erscheinen in Listen mit fett gesetzten Bezeichnungen.
  `\item[]` führt die Erläuterung eines Listenpunkts ohne neue Nummer fort.
- Trennlinien stehen zwischen größeren Blöcken, nicht zwischen jeder Regel
  oder jeder einzelnen Aufgabe. Wichtige Begriffe werden gezielt fett gesetzt.

## Verbindliche Schreibregeln

1. Keine Satzinterpunktion in abgesetzten Gleichungen: keine abschließenden
   Punkte, Kommata, Semikola oder Doppelpunkte, auch nicht am Ende einzelner
   Zeilen einer mehrzeiligen Rechnung. Mathematisch notwendige Zeichen wie
   Kommata in Koordinatentupeln und Funktionsargumenten bleiben erhalten.
   Normale Interpunktion im Fließtext bleibt erlaubt.
2. Keine Hervorhebung mit `\boxed{...}` oder vergleichbaren Formelkästen.
3. Als textlichen Trennstrich den einfachen ASCII-Bindestrich `-` verwenden,
   auch in Überschriften. Keine typographischen Gedankenstriche, kein `---`
   oder `--` als Ersatz. Mathematische Minuszeichen bleiben unverändert.
4. Inline-Mathematik vorzugsweise mit `$...$` schreiben. Abgesetzte Formeln
   in `equation` bzw. `equation*` statt `\[...\]` schreiben.
5. `equation` für zentrale Definitionen, allgemeine Rechenregeln und die
   maßgeblichen skalaren Aufgabenfunktionen verwenden. `equation*` für
   Beispielrechnungen, Seeds, Zwischenwerte und Hilfsdarstellungen verwenden.
   Maßgeblich ist die Auswahl im Referenzdokument; nicht jede Formel nummerieren.
6. Zusammengehörige Rechenschritte mit `split` innerhalb der gewählten
   Gleichungsumgebung bündeln; Gleichheitszeichen oder Implikationen mit `&`
   ausrichten. Eine gemeinsame Gleichungsnummer genügt für einen Regelblock.
7. Keine automatische Abschnittsnummerierung zusätzlich zu den expliziten
   Titeln: `\setcounter{secnumdepth}{0}` beibehalten. Innerhalb jedes Theorieteils
   beginnt die explizite Nummerierung der Subsections erneut bei 1.
8. `\section` für Theorieteile, Aufgabenblöcke, Abschlussfragen und „Key concepts“,
   `\subsection` für Theorieabschnitte und einzelne Aufgaben,
   `\subsubsection` für untergeordnete Beispielelemente verwenden.
   Bei Blättern ohne übergeordnete Teile (wie Übung 2) die einzelnen Übungen
   direkt als `\section{Exercise 1 - ...}`, `\section{Exercise 2 - ...}` usw.
   gliedern. Keine künstlichen „Part“-Abschnitte hinzufügen; erläuternde
   Unterabschnitte einer Übung können als `\subsection` gesetzt werden.
9. Kurze Abschlussfragen als gemeinsame `enumerate`-Liste statt mit einzelnen
   Überschriften setzen.
10. Aufgabenlisten fortlaufend nummerieren. Fortsetzungen mit `\item[]`
   erhalten keine zusätzliche Nummer. Englische Inhalte und mathematische
   Aussagen beibehalten; erläuternde Formulierungen dürfen gestrafft werden.

## Layout und Bearbeitung

- A4, 11 pt und Kurskopf aus `course_sheet.cls`; Schrift und Absatzabstände
  entsprechend `course-sheet.latex`. Der zusätzliche Zeilenumbruch vor dem
  Blatttitel trennt ihn vom Kurstitel.
- Keine Absatzeinzüge, moderater Absatzabstand, flexible untere Seitenkante.
- Tabellen mit umbrechenden Spalten innerhalb der Textbreite. Für kurze
  Tabellen eine einfache `tabular`-Umgebung mit `p{...}`-Spalten und
  `\toprule`, `\midrule`, `\bottomrule` verwenden. Keine unnötigen
  `minipage`-Umgebungen, Breitenberechnungen oder `longtable`-Kopfdefinitionen.
- Überschriften und anschließenden Inhalt zusammenhalten; nach Änderungen
  das PDF auf abgeschnittene Formeln, Tabellen und ungünstige Umbrüche prüfen.
- Markdown ist die maßgebliche Quelle. Die bisherigen Autorenänderungen aus
  den TeX-Prototypen wurden übernommen. TeX wird bei jeder Konvertierung neu
  erzeugt; dauerhafte Änderungen deshalb ausschließlich im Markdown vornehmen.
- Die Regeln sind für weitere Überarbeitungen vorgesehen. Andere Blätter
  und die Lösung werden dadurch nicht automatisch redaktionell angepasst.

## Markdown-Konventionen und Workflow

Der YAML-Kopf enthält `title` und `published`; für TikZ-Diagramme zusätzlich
`tikz: true`. `#`, `##` und `###` erzeugen Section, Subsection und Subsubsection.
Nummern von Teilen und Übungen bleiben explizit im Titel. Kurze Fragen und
Antworten als Markdown-Liste `1.`, `2.` usw. schreiben. Eingerückte Absätze
führen denselben Listenpunkt fort, ohne eine neue Nummer zu erzeugen.

Inline-Mathematik steht in `$...$`, abgesetzte Mathematik in `$$...$$`.
Standardmäßig wird daraus `equation*`. Für nummerierte Definitionen:

```markdown
::: numbered
$$f(x) = x^2$$
:::
```

Mehrzeilige Rechnungen behalten `\begin{split} ... \end{split}` innerhalb
von `$$...$$`. Interpunktion und Formelkästen bereits im Markdown vermeiden;
der Renderer verändert mathematische Ausdrücke nicht automatisch.

Tabellen als gewöhnliche Pipe-Tabellen schreiben. Bei Bedarf steuert ein
umgebender Div die Spaltenbreiten, zum Beispiel:

```markdown
::: {.course-table columns="@{}p{0.32\\linewidth}p{0.64\\linewidth}@{}" tabcolsep="3pt"}
| Concept | Meaning |
|---|---|
| Gradient | Direction of steepest increase |
:::
```

Ohne Angabe verwendet der Renderer gleich breite, umbrechende Spalten.
Die Ausgabe ist eine einfache `tabular`-Umgebung mit `booktabs`-Linien.
Ein zusätzlicher `::: center`-Div zentriert Tabellen oder Diagramme.
Bilder werden relativ zur Markdown-Datei referenziert, etwa `../img/trees_forest.pdf`.

Überschriften können `{needspace="8"}` tragen, um acht Zeilen Platz für
Überschrift und Folgeinhalt zu reservieren. Bewusste Seitenumbrüche sowie
TikZ-Diagramme bleiben als mit `{=latex}` markierte Codeblöcke erhalten.

`python3 scripts/build_attendance.py` rendert und kompiliert alle aktiven
Markdown-Dateien mit demselben Kurskopf. Der GitHub-Workflow verwendet diesen
Befehl. Die Ausgabe liegt unter `build/`; archivierte Dateien werden nicht
berücksichtigt. `scripts/render_tex.sh <datei.md>` erzeugt optional eine
TeX-Vorschau direkt neben dem Markdown und überschreibt vorhandene Vorschauen.
