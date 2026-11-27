# Cubeplanet – Gesamtplan: Spielkonzept, Engine-Neubau, Fahrplan

## Kontext

Cubeplanet ist heute ein Nachbau von Minecraft-Survival plus Create/Aeronautics auf **einem** Würfelplaneten: ~75 Maschinentypen, Fahrzeugphysik, Züge – aber kein Spielverlauf, keine Erze, kein Tech-Baum, keine Kreaturen, nur ein Planet. Die Review zeigte außerdem: Der Code ist auf genau einen Planeten und kleine Fabriken festgelegt (ein Riesenobjekt `Game`, absolute 32-Bit-Koordinaten im Renderer, jede Maschine wird jeden Takt voll berechnet).

**Dein Auftrag:** die gesamte Engine neu schreiben und zuerst ihre Grundsteine legen – so zukunftssicher wie möglich, auf **maximale Geschwindigkeit und maximale Schönheit** mit modernen Shadertechniken ausgelegt, mit allen Voxel-Techniken, die es erlauben, **ein ganzes Universum auf einmal zu sehen**, und mit Terrain, das sich auch auf einer einzelnen Planetenseite nicht wiederholt.

### Deine Entscheidungen
- **Engine-Neubau**, Altcode darf vollständig ersetzt werden.
- **Einzelspieler zuerst**, Koop später (Simulation deterministisch, vom Client getrennt).
- **Ein handgestaltetes Sonnensystem** (8 Körper), im Endgame erzeugte Nachbarsysteme.
- **Gefahren:** Wetter-Katastrophen, die töten, aber nichts zerstören; lebendige Tierwelt; lebensfeindliche Umgebungen. Erste Welten gemütlich, nach außen härter. Bosse später.
- **Später veröffentlichen:** fremde Grafiken nur als Platzhalter, eigene Inhalte austauschbar eingeplant.
- **Nur lokal arbeiten**, kein Push.

### Von mir gesetzt (bei der Freigabe widersprechbar)
- **Mindest-Hardware = dein Rechner:** GTX 1080, 1080p, 60 Bilder/s mit allen Effekten. Stärkere Karten bekommen Zusatzstufen.
- **1-Meter-Blöcke bleiben.** Kleinere Voxel kosten das Tausendfache an Speicher und passen nicht zum Blockbau.
- **Kompaktes Sonnensystem** (Mond ~120 km entfernt, Planeten einige tausend km auseinander), damit Nachbarwelten am Himmel als echte Körper zu sehen sind und Reisen Minuten dauern.
- **Rust + wgpu bleiben**, Vulkan als Hauptweg, DX12 als Ausweichweg.

---

# Teil A – Spielkonzept

### Leitidee
Du beginnst mit leeren Händen auf einer gemütlichen Würfelwelt. Jedes Zeitalter gibt dir **eine neue Energiequelle, eine neue Transportebene und eine neue Reichweite** – und automatisiert, was du im Zeitalter davor von Hand getan hast. Am Ende läuft ein Netz aus Fabriken über viele Welten, ohne dass du daneben stehst.

### Fünf Säulen
1. **Bauen ist Physik.** Masse, Schwerpunkt, Auftrieb und Schub zählen.
2. **Jedes Zeitalter spielt sich anders.**
3. **Planeten sind Charaktere:** eigene Gefahr, eigener Schlüsselrohstoff, eigene Tiere, andere Bauweise.
4. **Automatisierung wächst mit** – vom Wasserrad bis zum Standort, der ohne dich produziert.
5. **Gemütlich zuerst, härter weiter draußen.**

### Spielschleife

Vier ineinander liegende Schleifen. Jede endet damit, dass etwas **von allein läuft**, was vorher Handarbeit war – und genau dadurch entsteht der nächste Engpass.

**1. Momentschleife (30 Sekunden bis 5 Minuten)**
Fehlendes Teil sehen → Rohstoff holen → verarbeiten → einbauen → **sofort sichtbare Wirkung**.

**2. Sitzungsschleife (30 bis 90 Minuten)**
1. **Engpass wählen** – das Logbuch zeigt, was fehlt.
2. **Linie bauen**, die ihn löst.
3. **Linie läuft allein.**
4. **Überschuss fließt in Forschung** und schaltet einen Knoten im Tech-Baum frei.
5. **Neue Maschine oder Fähigkeit** verschiebt den Engpass: Transport, Energie oder ein Rohstoff, den es hier nicht gibt.

**3. Zeitalterschleife (5 bis 15 Stunden)**
Neue Energiequelle → neue Transportebene → **neue Reichweite** → Außenposten → automatisieren und anbinden → **Tor-Projekt** (Großbau als Meilenstein: Wasserrad, Dampfkessel, Generator, Rakete, Mondbasis, Warpkern) → nächstes Zeitalter.

**4. Planetenschleife (ab Raumfahrt)**
**Ankommen** → **Überleben** → **Zähmen** → **Automatisieren** → **Anbinden** → **Weiterziehen** (der Standort läuft per Fernsimulation weiter).

**Was die Schleife antreibt**
- **Logbuch:** erreichbare Tech-Knoten, laufendes Tor-Projekt, größter Engpass mit Durchsatz-Zahlen.
- **Senken für Überschuss:** Forschung, Tor-Projekte, Ausrüstung, Dekoration.
- **Rhythmus:** Tag/Nacht und Wetter zwingen kurz in den Schutzraum – dort plant und forscht man.
- **Kein Verlieren:** Die Fabrik geht nie kaputt. Beim Tod wacht man an der Basis auf; das Inventar wartet in einer Kiste am Unfallort.

**Die ersten Stunden**

| Zeit | Was passiert |
|---|---|
| 0–5 min | Feuerstein, Stöcke, Fasern; Faustkeil und Axt |
| 5–15 min | Unterstand, Lagerfeuer, erste Mahlzeit |
| 15–30 min | Lehm finden, Brennofen (erstes Tor-Projekt) |
| 30–60 min | Kupfer und Zinn, Bronze, bessere Werkzeuge |
| 1–2 h | Feld, Tier zähmen, Göpel treibt den ersten Mahlstein |
| 2–3 h | Wasserrad, Wellen, erstes Förderband |
| 3–5 h | Eisenlinie läuft automatisch; Dampfkessel als nächstes Tor |

### Zeitalter

| # | Zeitalter | Energie | Neue Transportebene | Schlüsselmaterial | Tor zum nächsten |
|---|---|---|---|---|---|
| 0 | Steinzeit | Muskelkraft, Feuer | Tragen, Schlitten | Feuerstein, Holz, Lehm | Brennofen, erstes Kupfer |
| 1 | Bronzezeit | Tierkraft (Göpel), Handkurbel | Karren, Lasttiere, Floß | Bronze, Keramik, Ackerbau | Wasserrad |
| 2 | Mechanik | Wasser, Wind | Förderbänder, Boote | Eisen, Messing | Dampfkessel |
| 3 | Dampf & Stahl | Dampf | Züge, Rohre, Luftschiffe, Flugzeuge | Stahl, Glas | Generator |
| 4 | Elektrizität | Strom | Stromnetz, Pakete, Autopilot-Routen | Kupferdraht, Silizium | Rakete im Orbit |
| 5 | Raumfahrt | Treibstoff, Solar | Frachtraketen mit Fahrplan | Titan, Aluminium | Mondbasis, die sich selbst versorgt |
| 6 | Interplanetar | Kernspaltung → Fusion | **Fusionstriebwerke**, Orbitalstationen, Massentreiber, Fern-Standorte | ein Schlüsselrohstoff je Planet (Deuterium von Thalassa, Helium von Zephyr als Fusionsbrennstoff) | Warpkern |
| 7 | Interstellar | Sternenenergie | **Warpantrieb** zu fremden Sternen, Baupläne + Bau-Drohnen | exotische Materie | offenes Endgame |

**Antriebe (Owner-Entscheidung vom 27.11.):** Zwischen den Planeten fliegt man mit Fusionstriebwerken, zu fremden Sternen erst ganz am Ende mit Warp. Sternentore gibt es nicht.

**Fortschritt:** sichtbarer Tech-Baum; Knoten kosten **Abgaben**, teils zusätzlich eine **Entdeckung** (Probe) oder **Leistung** (Flughöhe, Orbit).

### Sonnensystem (Arbeitstitel)

| Körper | Charakter | Gefahr (tötet, zerstört nichts) | Schutz | Schlüsselrohstoff | Stufe |
|---|---|---|---|---|---|
| **Heimat** | sechs Klimaseiten | nur Stimmungswetter | – | Kupfer, Zinn, Eisen, Kohle | gemütlich |
| **Kiesel** (Mond) | klein, luftlos | Vakuum | Raumanzug, dichte Räume | Titan, Silizium | ruhig |
| **Thalassa** | Ozeanwelt mit weiten, knietiefen Flachmeeren und tiefen Becken | **Riesige Gezeitenwellen wie in „Interstellar“** (hunderte Meter bis ~1 km hoch, von weitem wie Berge am Horizont, kommen in festem Rhythmus), Sturmfluten, Druck | Schiffe, U-Boote, abgedichtete Bunker, rechtzeitig abheben | Deuterium, Salze | mittel |
| **Glacia** | Eiswelt | **Schneestürme**, Kälte | Heizung, Schutzräume | Lithium, gefrorene Gase | mittel |
| **Zephyr** | schwebende Inseln | **Tornados**, Blitze | Verankerung, Flugkönnen | Helium, Wasserstoff | mittel |
| **Ignis** | Vulkanwelt | **Schwefelnebel, toxischer Regen** | Filter, überdachte Wege | Schwefel, Wolfram | hart |
| **Cinder** | sonnennah | **Sonneneruptionen**, **Meteoritenschauer** | Bunker, **Schildgeneratoren** | Sonnenenergie, seltene Metalle | sehr hart |
| **Relikt** | äußere Ruinenwelt | Strahlung, Meteoriten | Schilde | Warp-Technik aus den Ruinen | Endgame |

### Terrain ohne Wiederholung

Eine Seite der Heimat misst rund 41 × 41 km. Reines Rauschen sieht auf dieser Fläche überall gleich aus. Gelände entsteht deshalb **von groß nach klein**:

1. **Planet:** Entstehungsgeschichte (Alter, Vulkanismus, Wasseranteil, Einschläge) → Gebirgszüge, Becken, Küsten, über Würfelkanten hinweg passend.
2. **Seite = Klima.**
3. **Landschaft:** je Seite 12–20 benannte Landschaften mit eigener Geologie, Palette, Höhenprofil und eigenen Formen (Tafelberge, Karst, Fjorde, Dünenmeer). Zwei Landschaften teilen nie denselben Bausatz.
4. **Gewässernetz und Erosion:** einmal je Körper als Karte **berechnet** (Abfluss bis zum Meer, Erosion, Ablagerung) – Täler führen wirklich irgendwohin.
5. **Klima im Kleinen:** Temperatur, Feuchte, Höhe, Regenschatten bestimmen Biome und Übergänge.
6. **Wahrzeichen:** einmalige Großformen mit **Abstandsregel** – von jedem Punkt ist eines sichtbar, kein Typ zweimal in Sichtweite.
7. **Untergrund:** Schichten, Erzgänge und Höhlen folgen der Geologie der Landschaft.
8. **Kleinteile:** Wuchsvarianten, Findlinge, Bodenbewuchs; Texturen ohne Kachelmuster, Farbverlauf je Landschaft.

### Gefahren, Tiere, Später
- Jedes Ereignis hat **Vorwarnung, Dauer, Gegenmittel**; Bauten nehmen nie Schaden; Schutz durch Innenräume, Anzug, Kabine, Schildkuppel. Ereignisse dürfen der Fabrik nützen (Sturm treibt Windräder).
- Je Körper 3–5 Tierarten; Nutztiere liefern Zugkraft (Göpel). Raubtiere hält man mit Zaun, Licht, Feuer fern.
- **Später:** Bosse, Koop, eigene Grafiken, Handel.

### Optik statt Redstone (Owner-Idee vom 27.11.)

Signale und Logik laufen über **Licht** statt über eine magische Leitung.

- **Bausteine:**
  - Lichtquellen: Feuer, Lampe, Sonnenlicht über einen Spiegel.
  - Spiegel lenken um, halbdurchlässige Spiegel verzweigen, Linsen bündeln und erhöhen die Reichweite.
  - Farbfilter legen mehrere Kanäle in einen Strahl, Prismen trennen sie wieder.
  - Blenden sind Schalter (bewegt von Welle, Hebel oder Strom). Lichtsensoren wandeln Licht zurück in Kraft oder Strom.
  - Später Lichtleiter (Glasfaser) als Kabel durch Wände und Wetter.
- **Logik:** Zwei Blenden hintereinander ergeben UND, zwei Strahlen auf einem Sensor ODER, ein Strahl, der eine Blende schließt, NICHT. Ein Strahl im Spiegelkreis hält sich selbst und wird so zum Speicher.
- **Wetter als Gegenspieler:** Nebel, Schneesturm und Schwefelnebel schwächen Strahlen; Sonnenlicht gibt es nur am Tag.
- **Zeitalter:**
  - Brennglas zum Feuermachen in der Stein- oder Bronzezeit.
  - Bronzespiegel als Heliograph zwischen Außenposten in der Mechanik.
  - Glaslinsen, Fernrohr und Sternwarte mit Dampf & Stahl (Glas), Photozellen mit der Elektrizität.
  - Danach Laserverbindungen zwischen Planeten, Sonnenöfen auf Cinder und Weltraumteleskope.
- **Fernrohr:**
  - Gebaut aus Rohr, Objektiv und Okular; größere Linsen und längere Rohre zeigen mehr.
  - Man sieht die echte Welt. Die Engine wählt Detailstufen nach Pixelgröße, eine Ansicht mit kleinem Blickwinkel holt ferne Körper also von selbst genauer heran: echte Berge auf dem Mond, Wolken auf Thalassa.
  - Sterne sind echte Systeme an ihren echten Orten (Sternkatalog). Großzügiger als die Wirklichkeit: Große Teleskope zeigen auch Planeten anderer Sterne.
- **Nutzen im Spiel:** Entdeckungen für den Tech-Baum, zum Beispiel den Mond kartieren, Rohstoffe und Landeplätze auf Nachbarwelten finden, Gefahren kommen sehen und später Ziele für Warpflüge finden.
- **Technik:**
  - Strahlen laufen ganzzahlig über das Blockgitter (deterministisch, mehrspielertauglich) und werden nur neu berechnet, wenn sich ein Block auf ihrem Weg ändert.
  - Ein Fernrohr ist eine zweite Kamera mit eigener Brennweite. Deshalb muss schon die Detailstufen-Auswahl in E3 mehrere Ansichten mit eigener Brennweite unterstützen.

---

# Teil B – Engine-Neubau

## B1. Grundsätze

1. **Alles hat Detailstufen:** Geometrie, Simulation und Speicher – nichts kostet nach Gesamtgröße der Welt, nur nach dem, was sichtbar oder aktiv ist.
2. **Kein fest eingebauter Planet:** Körper, Systeme, Blöcke, Materialien sind Daten.
3. **Simulation ohne Fenster lauffähig und deterministisch;** der Client zeigt nur an (Grundlage für Koop und Wiederholungs-Aufzeichnungen).
4. **Die Grafikkarte entscheidet, was gezeichnet wird** (GPU-getrieben); der Bild-Thread wartet nie auf Erzeugung oder Speicher.
5. **Messbar von Anfang an:** Benchmarks, Zeitmessung je Bildschritt und Vergleichsbilder gehören zur Engine, nicht nachträglich dazu.
6. **Ein Voxel-Weg für alles:** Gelände, Bauwerke und bewegte Konstruktionen nutzen dasselbe Speicher-, Mesh- und Zeichenformat.

## B2. Aufbau

Neuer Ordner `engine/` als eigener Cargo-Workspace. Der Altcode bleibt unangetastet und spielbar (Git-Marke `prototyp-v0`) und dient als Steinbruch.

| Crate | Aufgabe |
|---|---|
| `cp-core` | Mathematik, IDs, Bezugssysteme (Galaxie → System → Körper → Chunk), fester Takt, deterministischer Zufall |
| `cp-jobs` | Aufgabensystem mit Prioritäten, Abbruch und Zeitbudgets |
| `cp-content` | Datenregister (Blöcke, Materialien, Körper) mit Text-Schlüsseln, Neuladen zur Laufzeit |
| `cp-voxel` | Chunk-Speicher, Änderungen, Detail-Pyramide, Abfragen (Strahl, Kollision) |
| `cp-worldgen` | Generator-Schnittstelle mit Detailstufen, Makro-Karten, Terrain-Pipeline |
| `cp-universe` | Körper, Bahnen, Rotation, Sternkatalog, Zeit |
| `cp-mesh` | Mesh-Erzeugung (binäres Greedy-Meshing → gepackte Quads) |
| `cp-stream` | Detailstufen-Auswahl, Nachladen, Cache, Upload-Budgets |
| `cp-render` | Renderer, Bildschritt-Graph, alle Shader |
| `cp-sim` | Simulationskern: Befehle, Takt, Spielerkörper, Schwerkraft |
| `cp-save` | Regionsdateien, Versionen, Migrationen |
| `cp-app` | Programm: Fenster, Eingabe, Debug-Anzeigen, Betrieb ohne Fenster |
| `cp-tools` | Fähigkeiten-Abfrage der Grafikkarte, Benchmark, Bildvergleich, Karten-Backen |

**Wird portiert** (reine, gut geprüfte Mathematik): Schwerkraft-Potential (`src/physics/mod.rs`), Spielerbewegung (`src/physics/player.rs`), Bahn-Integration, Baum- und Höhlenformen (`src/world/generation/trees.rs`, `caves.rs`), Ideen aus `render/lod.rs`. **Wird nicht portiert:** `Game`, Maschinen, Speicherformat, Renderer.

## B3. Voxel-Techniken – Urteil

| Technik | Urteil | Grund |
|---|---|---|
| Erzeugte Basis + gespeicherte Änderungen | **übernehmen** | nur Änderungen kosten Speicher |
| Paletten-Chunks 32³, bitgepackt, einfarbige als ein Wert | **übernehmen** | tausende Blockzustände bei wenig Speicher |
| Detail-Pyramide aus echten Voxeln je Chunk | **übernehmen** | Bauwerke und Krater sind auf jeder Entfernung korrekt |
| Octree-Detailstufen je Körper nach Pixelgröße | **übernehmen** | gleiche Dreieckszahl vom Boden bis zum Orbit |
| Blockige Fernstufen (ein Mesh-Format überall) | **übernehmen** | ein Weg, kein Formwechsel an Bauwerken |
| Glatte Fernflächen (Marching Tetrahedra, Transvoxel) | verwerfen | zweite Pipeline, sichtbarer Formwechsel |
| Binäres Greedy-Meshing | **übernehmen** | Größenordnung 0,1 ms je Chunk |
| Gepackte Quads (8 Byte) + Vertex Pulling | **übernehmen** | rund 6-mal weniger Grafikspeicher als klassische Vertices |
| GPU-Culling (Sichtkegel, Rückseiten-Gruppen, Hi-Z-Verdeckung) + Multi-Draw-Indirect | **übernehmen** | CPU-Aufwand unabhängig von der Chunk-Zahl |
| Kamerarelative Koordinaten, 64-Bit-Bezugssysteme | **übernehmen** | millimetergenau auf jedem Körper |
| Umgekehrte Tiefe mit unendlicher Fernebene | **übernehmen** | ein Tiefenpuffer vom Block bis zum Nachbarplaneten |
| Logarithmische Tiefe, getrennte Fern-Durchgänge | verwerfen | unnötig mit umgekehrter Tiefe |
| Gebackene Makro-Karten je Würfelseite | **übernehmen** | speisen Generator, Fern-Schattierung, Fernschatten |
| Generator mit Detailstufen und Min/Max-Grenzen | **übernehmen** | leere und volle Bereiche werden übersprungen |
| Lichtvolumen (Himmel + farbiges Blocklicht) als 3D-Textur | **übernehmen** | weiches Licht in Höhlen für 0,2 ms |
| Belegungs-Clipmap auf der GPU | **übernehmen** | Regenverdeckung, Umgebungsverdeckung, Basis für Raytracing |
| Voxel-Raytracing für Schatten und indirektes Licht | später | Zusatzstufe für stärkere Karten |
| Alles per Raymarching (Teardown-Stil) | verwerfen | für texturierte 1-m-Blöcke auf der GTX 1080 langsamer |
| SVO / SVDAG als Hauptspeicher | verwerfen | Änderungen zu teuer |
| Brickmaps / 64-Trees | später | GPU-Struktur für die Raytracing-Stufe |
| Punkt-Darstellung ferner Körper unter 2 Pixel | **übernehmen** | darüber echte Geometrie aus den gröbsten Stufen |
| Randwände + geditherte Überblendung an Stufengrenzen | **übernehmen** | keine Risse, kein Aufploppen |
| Meshlets (Nanite-artig), Splats | verwerfen | brauchen Mesh-Shader bzw. passen nicht zu Blöcken |
| Regionsdateien, inhaltsadressierter Cache in Pack-Dateien | **übernehmen** | ersetzt die 87.000 Einzeldateien |
| CSG-Stempel für Großeingriffe (Krater) | später | Speicherformat sieht Operationen vor |

Geprüft: Die verwendete wgpu-Version 30.0.1 bietet Multi-Draw-Indirect (auch mit Zähler), Zeitstempel-Abfragen, 11/10-Bit-HDR-Ziele und BC-Kompression an. Was die GTX 1080 davon wirklich meldet, misst der erste Meilenstein.

## B4. Maßstabsleiter – das Universum auf einmal

| Entfernung | Darstellung |
|---|---|
| 0–300 m | exakte Blöcke, volle Texturen, Lichtvolumen, Detailpflanzen |
| bis zum Horizont | Detailstufen 1–11 desselben Voxel-Octrees; Fernflächen mit Farbe und Normalen aus den Makro-Karten beleuchtet |
| anderer Körper am Himmel | sein eigener Octree, nur die gröbsten Stufen; eigene Atmosphäre, Phasen, Finsternisse |
| Körper unter 2 Pixel | leuchtender Punkt mit korrekter Helligkeit |
| andere Sterne | Punkte aus dem Sternkatalog – jeder ist ein besuchbares System an seiner echten Position |

Übergänge sind stufenlos: Jeder Körper wählt seine Detailstufe nur nach Pixelgröße. Eine **Karten-Kamera** zoomt ohne Schnitt vom Block über Planet und System bis zur Galaxie.

## B5. Bild-Pipeline (moderne Shadertechniken)

Zielbudget auf der GTX 1080 bei 1080p: **13,4 ms** (74 Bilder/s); dynamische Auflösung hält 60.

| Schritt | Technik | Ziel |
|---|---|---|
| Sichtbarkeit | GPU-Culling, zweistufige Hi-Z-Verdeckung | 0,3 ms |
| G-Buffer | gepackte Quads, kompakt (Farbe, Normale + Rauheit, Material, Bewegung) | 2,0 ms |
| Schatten | 4 Kaskaden, weiche Kanten, Kontaktschatten; Fernschatten über die Höhenkarte | 2,5 ms |
| Licht | physikalisch basiert, Lichtvolumen, GTAO, Wolkenschatten | 2,0 ms |
| Atmosphäre | physikalische Streuung mit Tabellen, an die Würfelform angepasst, vom Boden bis in den Raum | 0,8 ms |
| Volumetrik | Froxel-Nebel mit Lichtschächten; trägt später Schneesturm und Schwefelnebel | 1,0 ms |
| Wolken | volumetrisch, Viertelauflösung, zeitlich rekonstruiert | 1,5 ms |
| Wasser | Wellen, Brechung, Spiegelung, Schaum, Kaustik | 1,0 ms |
| Spiegelungen | Screen-Space, halbe Auflösung | 0,8 ms |
| Nachbearbeitung | TAA mit optionaler Hochskalierung, Bloom, Auto-Belichtung, AgX-Tonemapping, Schärfen | 1,5 ms |

Dazu in späteren Grundsteinen: Nässe und Schneedecke je nach Himmelssicht, GPU-Partikel (Regen, Schnee), Gras und Blätter als Detail-Instanzen mit Wind. **Texturen:** eigener prozeduraler 16-Pixel-Satz als Standard (frei von fremden Rechten); die vorhandenen Platzhalter-Texturen sind zuschaltbar und als solche markiert.

## B6. Geschwindigkeitsziele (werden gemessen, nicht versprochen)

- Abstiegsflug Orbit → Boden: P99 unter 16,6 ms, kein Einzelbild über 20 ms beim Nachladen.
- Mesh-Erzeugung im Schnitt unter 0,2 ms je Chunk, Gelände-Erzeugung unter 3 ms je Chunk.
- Blockänderung nach spätestens 2 Bildern sichtbar.
- Grafikspeicher unter 3 GB.
- Vergleich: nicht langsamer als der Prototyp auf derselben Strecke – bei deutlich mehr Bildqualität.

## B7. Zukunftssicherheit

- Spielstände von Anfang an mit Namens-Palette und Versions-Migrationen.
- Fähigkeiten der Grafikkarte werden beim Start ausgehandelt; Bildschritte sind austauschbar (Raytracing-Stufe später).
- Keine fremden Engine-Frameworks; schlanke eigene Schichten auf wgpu und winit.
- Kern-Crates ohne Windows-Abhängigkeit; Tests laufen auch ohne Grafikkarte.

---

# Teil C – Fahrplan

## Phase I – Engine-Grundsteine

Jeder Grundstein endet mit einem lauffähigen Programm und einer messbaren Abnahme.

| # | Grundstein | Abnahme |
|---|---|---|
| **E0** | Gerüst: Workspace, Kern-Typen, Aufgabensystem, Register, Fähigkeiten-Abfrage | Bericht der echten GTX-1080-Fähigkeiten; alle Prüfbefehle grün |
| **E1** | Voxel-Kern: Paletten-Chunks, Detail-Pyramide, Abfragen, Greedy-Mesher | Mesh identisch zur einfachen Vergleichslösung; Zeit je Chunk gemessen |
| **E2** | Renderer-Kern: Bildschritt-Graph, GPU-getriebene Voxel-Pipeline, G-Buffer, Debug-Ansichten, Zeitmessung | Testwelt mit mehreren tausend Chunks; Bild ohne Fenster als PNG |
| **E3** | Nachladen und Detailstufen: Octree, Aufgaben, Cache, Überblendung | nahtloser Flug vom Block in den Orbit eines 41-km-Würfels; Benchmark-Strecke |
| **E4** | Universum: mehrere Körper, Bahnen, Rotation, Sternkatalog, Karten-Kamera | auf der Heimat stehen, Mond und Planeten sehen; ohne Ladebild hinfliegen; Zoom bis zur Galaxie |
| **E5** | Licht und Atmosphäre: Schatten, Lichtvolumen, GTAO, Atmosphäre, HDR, TAA, Tonemapping | Vergleichsbilder (Mittag, Dämmerung, Nacht, Höhle, Orbit); Budget gemessen |
| **E6** | Terrain 2.0: Makro-Karten, Erosion, Landschaften, Wahrzeichen, Bäume und Höhlen | Wiederholungstest besteht; Wahrzeichen-Abstandsregel als Test |
| **E7** | Wolken, Wasser, Volumetrik, Spiegelungen | Vergleichsbilder (Küste, Wasserfall, Gewitterhimmel, Planet aus dem Orbit) |
| **E8** | Simulationskern: Befehle, Takt, Spielerkörper, Kollision, Bauen/Abbauen, Speichern | laufen, graben, bauen auf allen sechs Seiten; Spielstand übersteht Neustart |
| **E9** | Wetter und Detail: Partikel, Nässe, Schnee, Gras mit Wind | Schneesturm- und Regenszene als Vergleichsbild im Budget |

## Phase II – Das Spiel auf der neuen Engine

1. **Baukasten:** Items, Rezepte und Maschinen als Daten; Netze (Kraft, Rohre, Strom, Logistik, Licht für Signale) als eigene Objekte mit räumlichem Index und Ruhezustand.
2. **Erste Spielschleife:** Zeitalter 0–2 auf der Heimat mit Tech-Baum, Logbuch, Tor-Projekten.
3. **Fabrik-Skalierung und Fernsimulation.**
4. **Wetter-Gefahren, Innenräume, Tiere.**
5. **Dampf und Strom.**
6. **Raumfahrt:** bewegte Konstruktionen mit Physik, Raketen, Mond.
7. **Planeten ausrollen**, dann **Interstellar**, dann **Veröffentlichungsreife**.

Die Create-Mechaniken des Prototyps kehren dabei als **eigene** Maschinen zurück, nicht als Kopie.

**Ehrliche Einordnung:** Phase I ist ein großes Stück Arbeit über viele Arbeitsgänge; Phase II dauert Jahre. Die Fehler des Prototyps behebe ich nicht mehr – sie werden zu Anforderungen und Tests der neuen Engine.

