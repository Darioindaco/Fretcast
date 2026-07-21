# Fase 0 — Opnameprotocol (eval-harnas voeden)

Doel: een gelabelde persoonlijke dataset op **jouw** gitaar + DI-setup, **zonder ook
maar één noot handmatig te annoteren**. Je speelt voorgeschreven tabs op een klik; de
ground truth is de tab zelf, de onset-tijd volgt uit de maat (`beat × 60/bpm`).

## Setup (eenmalig)

- Elektrische gitaar → interface, **droge DI**. Neem het DI-signaal op, niet de amp-sim.
- Verse snaren, normale actie, standaard stemming (E A D G B E). Stem strak — de
  ground truth gaat uit van exacte stemming.
- Klik/metronoom op **80 BPM**, hoorbaar in je monitoring maar **niet** in de opname
  (aparte track of mute op de opnamebus). Eén-maats count-in, dan starten de exercises
  op de downbeat = beat 0.
- Sample rate 44.1/48 kHz, 24-bit, mono DI. Eén WAV per exercise, benoemd
  `ex1_chromatic.wav`, `ex2_pentatonic.wav`, `ex3_devil_pairs.wav`.

## De drie exercises

| # | Bestand | Wat | Meet |
|---|---------|-----|------|
| 1 | `ex1_chromatic.csv` | Frets 1-4 op elke snaar, laag E → hoog E | Rauwe string+fret-accuraatheid, geen ambiguïteit |
| 2 | `ex2_pentatonic.csv` | A-mineur pentatonisch, 5e-positie box op en neer | String+fret in een muzikale frase; voedt de positie-prior |
| 3 | `ex3_devil_pairs.csv` | Zelfde toonhoogte op twee posities (ver uit elkaar) | `disambiguation_rate` — het getal dat de kern van V02 is |

Speel elk exact op de klik, schone toon, noten netjes gescheiden (fingerpicking-stijl).
Fout gespeeld? Neem die exercise opnieuw op — de tab is de waarheid, niet je take.

## Uitbreiden (aanrader zodra de baseline draait)

- Herhaal alle drie **3-5×** met kleine variatie (aanslagdynamiek, andere pickup-stand):
  dit is je augmentatie-vrije domain-data en verdubbelt goedkoop de trainset (plan §7,
  "goedkoopste knop").
- Voeg later een `ex4_fingerpicking.csv` toe met een echt fragment uit je repertoire
  (Elliott Smith / Frusciante-hoek) — dat is je realistische testset.

## Valideren zonder model

Zodra de WAV's er zijn en een model tab uitspuugt in `time_sec,string,fret`-formaat:

```
python core/scripts/run_baseline.py --exercise ex1_chromatic --wav path/to/ex1_chromatic.wav
```

Dat script draait het model, schrijft de voorspelling weg en print de eval tegen de
ground-truth CSV. Vóór er een model is, bewijst `python core/tests/test_tab_eval.py`
dat de wiskunde en het harnas correct meten.
