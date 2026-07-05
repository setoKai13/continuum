# Continuum

Un agent qui garde le fil.

Continuum est un agent de poste macOS declenche a la voix : il voit l'ecran,
agit (souris, clavier) et tient un **etat de tache persistant** (hold-state).
Il ne repart pas d'un instantane fige : il reprend une tache longue la ou elle
s'etait arretee, et integre les **corrections dites a la voix** en cours de
route (les etapes restantes sont recalibrees, une etape bloquee retrouve sa
chance sous la nouvelle regle).

## Comment ca marche

Maintiens F8, parle, relache : l'instruction est transcrite en local
(faster-whisper), decomposee en etapes par Gemini, puis la boucle
OBSERVE -> UPDATE_STATE -> DECIDE -> ACT avance etape par etape — screenshot,
decision (fast-path zero-LLM ou grounding vision Gemini), action reelle
(pyautogui), verification sur le nouvel ecran. Esc met en pause ;
`--resume <task_id>` reprend exactement ou on s'etait arrete. Un HUD terminal
montre l'etat de tache et chaque action en direct.

## Prerequis

- macOS
- Python 3.11+
- Une cle d'API Gemini (voir `.env.example`)

## Installation

```bash
cd python-desktop
cp .env.example .env               # renseigner GEMINI_API_KEY
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Tester sans cle (headless)

```bash
.venv/bin/pytest -q                 # suite complete
.venv/bin/python scripts/dry_run.py # 8 scenarios de preuve, "DRY-RUN OK"
```

## Lancer en live

```bash
.venv/bin/python main.py                          # nouvelle tache (parle au F8)
.venv/bin/python main.py --resume <task_id>       # reprendre une tache
```

Details (permissions macOS, telechargement du modele STT, touches, depannage) :
`python-desktop/RUNBOOK.md`. Architecture : `ARCHITECTURE.md`.

## Licence

MIT.
