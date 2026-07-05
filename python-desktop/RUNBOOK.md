# RUNBOOK -- Continuum (Python, hold-state Mac agent)

## 0. Quickstart equipe -- cloner et tester en 2 minutes

```bash
git clone <URL_REPO> && cd <repo>/python-desktop
cp .env.example .env                        # puis mets ta cle dans GEMINI_API_KEY=
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # set complet live (GUI/audio/Gemini)

# Sans cle / sans Mac configure (headless), tout se prouve deja :
.venv/bin/pytest -q                         # suite complete, 0 failed attendu
.venv/bin/python scripts/dry_run.py         # 8 scenarios, affiche "DRY-RUN OK"

# Avec cle + permissions macOS (section 2) :
.venv/bin/python main.py                    # parle au F8, Esc pour pauser
```

Pour du dev headless pur (tests, dry-run, `import main`), le set minimal
suffit : `.venv/bin/pip install pydantic pydantic-settings pytest rich`.

This build was produced headless (no real Gemini call, no real click, no
microphone). Everything mechanical is proven by tests and
`scripts/dry_run.py`. The items below marked **TODO(tom)** are the only
things a human on the actual Mac needs to do before a live demo.

## 1. Setup (already done by the build, listed for reference)

```bash
cd python-desktop
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # full live set (GUI/audio/Gemini)
```

## 2. TODO(tom) -- before any live run

1. **Real Gemini key.** Open `.env` and replace the placeholder:
   ```
   GEMINI_API_KEY=REPLACE_WITH_REAL_KEY_TODO_TOM
   ```
   with your real key from the DeepMind hackathon Google Form. Without
   this, `vision.py` raises `MissingApiKeyError` on first `ground()` call
   (checked via `config.is_placeholder_key`, so it fails loud, not silently).

   **CRITIQUE (verifie 2026-07-05) :** depuis le 2026-06-19 l'API Gemini
   **rejette les cles standard NON restreintes** (403 / PERMISSION_DENIED).
   Teste la cle du stand AVANT la demo :
   ```bash
   .venv/bin/python -c "from google import genai; \
     print(genai.Client(api_key='<CLE>').models.generate_content(\
     model='gemini-3.5-flash', contents='ping').text)"
   ```
   Si 403 : regenere une cle depuis AI Studio (auto-restreinte) ou ajoute une
   restriction manuelle. Voir `../CONNECTEURS-VERIF-2026-07-05.md`.
   Le modele par defaut est `gemini-3.5-flash` (Computer Use natif) ; fallback
   `gemini-2.5-flash` ou legacy `gemini-2.5-computer-use-preview-10-2025` via
   `MODEL_NAME` dans `.env`. Option a tester : l'environnement Computer Use
   `DESKTOP` (Voie 1) peut viser plus juste que le grounding-vision par defaut.
2. **Install the full dependency set** on the demo Mac (not just the
   minimal gate set): `pip install -r requirements.txt`.
3. **Whisper model download (~460 MB, one-time)**: the STT model
   (`WHISPER_MODEL`, default "small") downloads from Hugging Face on first
   load. Pre-download it on good wifi BEFORE the demo (it is then cached in
   `~/.cache/huggingface`):
   ```bash
   .venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('small', compute_type='int8'); print('cached OK')"
   ```
   The model also pre-loads at agent startup (`PushToTalkListener.start()`
   warms it up), so the first F8 press is never the one paying the load.
4. **Grant 4 macOS permissions** to your terminal / python binary (System
   Settings -> Privacy & Security), then **relaunch the terminal**:
   - **Accessibility** (pyautogui clicking/typing)
   - **Input Monitoring** (pynput global F8/Esc key listening -- distinct
     from Accessibility on macOS 10.15+; without it the hotkeys are
     silently dead)
   - **Screen Recording** (mss screenshot capture; without it macOS
     returns wallpaper-only frames, no error)
   - **Microphone** (sounddevice push-to-talk recording; the OS prompt
     fires on the first real recording)
   `main.py` calls `mac_control.check_macos_permissions()` at boot:
   Accessibility and Screen Recording use the real CoreGraphics preflight
   checks and exit with a clear error if missing. The Microphone probe is
   best-effort and Input Monitoring has no cheap preflight -- test both
   with the dry pass below.
5. **F8 is a media key on a default Mac**: out of the box, pressing F8
   sends Play/Pause, NOT the F8 keycode -- the push-to-talk would be dead
   AND music could start mid-demo. Either check "Use F1, F2, etc. keys as
   standard function keys" (System Settings -> Keyboard), or hold Fn+F8,
   or set another key via `PTT_KEY` in `.env`.
6. **Real click / real micro test**: before the demo, do one dry pass --
   hold F8 (default `PTT_KEY`), say a short command, release, and confirm
   the HUD shows a transcript and the agent acts on the real screen.
   Logs go to `continuum.log` (`LOG_PATH`), not the console (the console
   belongs to the HUD): `tail -f continuum.log` in a second terminal.
   Emergency stops, in order: `KILL_KEY` (Esc), mouse slammed into any
   screen corner (pyautogui fail-safe, honored as kill-switch), Ctrl+C.

## 3. Live launch commands

Start a brand-new task:

```bash
.venv/bin/python main.py --goal "Triage the Linear queue"
```

Resume a previously paused/active task (proves hold-state: reloads the
`TaskState` from `continuum.db`, bumps `session_count`, restarts at the
first non-done step instead of from scratch):

```bash
.venv/bin/python main.py --resume TRI-3
```

While running:
- Hold `PTT_KEY` (default `f8`) to speak an instruction; release to
  transcribe and feed it into the OBSERVE step.
- **Correct the agent by voice, mid-task**: a phrase carrying a correction
  marker ("non, ...", "en fait ...", "actually ...", "plutot ...") is
  detected as an override -- Gemini extracts the (when, rule) pair, every
  remaining matching step is recalibrated, and a step previously blocked
  under the old rule returns to `todo` with a fresh attempt budget. The
  agent confirms out loud ("Correction noted: ...") and the override
  appears in the HUD state panel.
- Press `KILL_KEY` (default `esc`) to trigger the kill-switch: the loop
  pauses within one turn and persists `status=paused` (resumable next
  session).
- The left HUD panel shows the live `TaskState` (steps todo/doing/done,
  facts, overrides); the right panel streams every tool call.

## 4. Mechanical validation (headless, run it yourself)

```bash
.venv/bin/pytest -q                        # full suite, 0 failed expected
.venv/bin/python scripts/dry_run.py        # 8 scenarios, prints "DRY-RUN OK"
.venv/bin/python -c "import main"          # imports clean without GUI/API deps
```

## 5. What the code does (module map)

- `state.py` : le hold-state (TaskState, mark_step, apply_override,
  next_actionable_step, resume, render) -- la primitive porteuse.
- `memory.py` : persistance SQLite (task_state/memory/task_log/trajectories),
  `load_startup_context`, `resume_task_state`.
- `agent.py` : boucle OBSERVE->UPDATE_STATE->DECIDE->ACT avec injection de
  dependances, refus destructif (2 gates), kill-switch, causalite, override
  live (correction vocale mid-task -> recalibrage des etapes restantes),
  budget de tours travail/attente separes.
- `vision.py` : planner/grounding/verify/override Gemini avec timeout +
  retry (transient seulement) + circuit breaker, parsing JSON defensif.
- `router.py` : blocklist destructive + fast-paths zero-LLM (open app/URL)
  + detecteur de marqueurs de correction.
- `mac_control.py` : denormalisation 0-1000 -> pixel (clampee dans l'ecran),
  capture Retina-safe, clic/type/hotkey/scroll, preflight permissions.
- `stt.py` / `tts.py` : push-to-talk faster-whisper local / synthese `say`.
- `hud.py` : HUD terminal Rich (etat de tache + stream d'actions).
- `main.py` : cablage live complet + `--resume` + degradation propre des
  erreurs Gemini (stall, jamais crash).
- `scripts/dry_run.py` : preuve headless des 8 invariants du produit.

## 6. Known simplifications (documented, not hidden)

- The live prompts (`vision.py`) are deliberately simple; tune them against
  the real demo app once the key is live.
- `check_macos_permissions()`: Accessibility and Screen Recording use real
  CoreGraphics preflights; the Microphone probe is best-effort and Input
  Monitoring is not probed (see section 2).
- `Speaker.say` is synchronous: a spoken confirmation pauses the loop for
  the duration of the sentence (~1-2s).
- The kill-switch is polled at the top of each turn: a turn stuck inside a
  slow Gemini call honors Esc only when the call returns (timeout 30s per
  attempt) -- the fail-safe corner and Ctrl+C remain instant.
