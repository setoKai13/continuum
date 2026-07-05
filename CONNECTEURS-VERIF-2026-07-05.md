# Connecteurs - etat verifie au 2026-07-05 (sources en ligne, datees)

Verification jour J des connecteurs utilises par Continuum : SDK Gemini, modeles,
Computer Use, Nemotron (OpenRouter), Gradium (voix). Chaque point est source et date.

---

## 1. Gemini SDK (google-genai / GEMINI_API_KEY)

**Statut : stable, avec 1 alerte critique cle API**

- `google-genai` est le SDK courant, en **GA** ("General Availability across all
  supported platforms"). `from google import genai` fonctionne.
  Source : https://ai.google.dev/gemini-api/docs/migrate (fetch 2026-07-05) ; https://pypi.org/project/google-genai/
- L'ancien SDK `google-generativeai` est **totalement en fin de support depuis le
  30 novembre 2025**. Le repo officiel s'appelle desormais
  `google-gemini/deprecated-generative-ai-python`. Ne pas l'utiliser.
  Source : https://github.com/google-gemini/deprecated-generative-ai-python (consulte 2026-07-05).
- `GEMINI_API_KEY` vs `GOOGLE_API_KEY` : "Set the environment variable GEMINI_API_KEY
  or GOOGLE_API_KEY... If both are set, GOOGLE_API_KEY takes precedence."
  Source : https://ai.google.dev/gemini-api/docs/api-key (fetch 2026-07-05).

### ALERTE CRITIQUE cle API (bloquante pour la demo)

Calendrier de securite des cles annonce par Google :
- **19 juin 2026** (deja passe) : l'API Gemini **rejette les cles "standard" non restreintes**.
- **Septembre 2026** : l'API rejettera **toutes** les cles "standard" (migration vers les "auth keys").
Sources : https://ai.google.dev/gemini-api/docs/api-key ;
https://discuss.ai.google.dev/t/action-required-restrict-gemini-api-keys-by-june-19-to-avoid-service-disruption/171786 ;
https://cybernews.com/security/google-gemini-reject-unrestricted-standard-keys/ (consultes 2026-07-05).

**Consequence pratique** : la cle utilisee pour la demo doit etre soit une "auth key"
(les nouvelles cles creees dans AI Studio le sont automatiquement), soit une cle
"standard" **restreinte**. Une cle standard non restreinte plante au premier appel
(403 / PERMISSION_DENIED). **Tester la cle des reception** :

```bash
.venv/bin/python -c "from google import genai; \
  print(genai.Client(api_key='<CLE>').models.generate_content(\
  model='gemini-3.5-flash', contents='ping').text)"
```

---

## 2. Modeles Gemini

**Statut : `gemini-3.5-flash` est GA et recommande pour Computer Use**

- `gemini-2.5-flash` existe toujours au catalogue, statut GA.
- `gemini-3.5-flash` est sorti le **19 mai 2026** (Google I/O 2026), **GA**,
  positionne "most intelligent... for agentic and coding tasks".
  Sources : https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/ ;
  https://www.marktechpost.com/2026/05/20/google-introduces-gemini-3-5-flash-at-i-o-2026-a-faster-and-cheaper-model-for-ai-agents-and-coding/ ;
  https://ai.google.dev/gemini-api/docs/models (page "Last Updated 2026-06-30").
- **Computer Use : depuis le 24 juin 2026**, ce n'est plus un modele a part mais une
  capacite integree a `gemini-3.5-flash`.
  Source : https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-computer-use-gemini-3-5-flash/ ;
  https://ai.google.dev/gemini-api/docs/interactions/computer-use ("Last updated 2026-06-25").
- Modeles supportant Computer Use (ordre de preference officiel) :
  1. `gemini-3.5-flash` -> **RECOMMANDE** (le defaut de `MODEL_NAME` dans `.env.example`)
  2. `gemini-3-flash-preview` -> preview
  3. `gemini-2.5-computer-use-preview-10-2025` -> **legacy**, fallback possible.

---

## 3. Computer Use : API, environnements, coordonnees

**Statut : l'environnement "desktop" (OS-level) existe officiellement**

Deux voies pour piloter le Mac :

- **Voie 1 - Interactions API** (`client.interactions.create(...)`), confirmee sur la
  doc officielle. Depuis l'integration a `gemini-3.5-flash` (24/06/2026), trois
  environnements officiellement supportes :
  - `ENVIRONMENT_BROWSER`
  - `ENVIRONMENT_MOBILE` (Android)
  - `ENVIRONMENT_DESKTOP` (**controle OS-level du curseur** : click, type, hotkey, scroll, drag_and_drop)
  Coordonnees : normalisees **0-999**, format `{x, y}` (pas une box).
  Sources : https://ai.google.dev/gemini-api/docs/interactions/computer-use (fetch 2026-07-05) ;
  https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-computer-use-gemini-3-5-flash/
- **Voie 2 - vision grounding via `generate_content`** (celle implementee dans
  `python-desktop/vision.py`) : screenshot + prompt -> le modele renvoie une box
  `[ymin, xmin, ymax, xmax]` normalisee **0-1000** (convention officielle Gemini
  image understanding, coin haut-gauche = origine), que `mac_control.py`
  denormalise et clique via pyautogui.
  Source : https://ai.google.dev/gemini-api/docs/image-understanding (confirme 2026-07-05).

**A tester en priorite le jour J** : Voie 1 avec `environment: "desktop"` directement
sur le Mac (ce n'etait pas possible avant fin juin 2026). Garder la Voie 2 (implementee,
testee headless) comme plan solide si la Voie 1 desktop se revele instable en pratique.

---

## 4. Nemotron (OpenRouter) - selecteur bonus

**Statut : confirme**

- `nvidia/nemotron-3-super-120b-a12b` existe au catalogue OpenRouter, avec suffixe
  `:free` disponible. Sorti le 11 mars 2026, 120B parametres / 12B actifs (MoE hybride
  Mamba-Transformer).
  Source : https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free (fetch 2026-07-05) ;
  https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models
- Base `https://openrouter.ai/api/v1`, compatible OpenAI SDK, cle via `OPENROUTER_API_KEY`.

---

## 5. Gradium (voix sponsor) - alternative STT/TTS

**Statut : confirme point par point via docs.gradium.ai**

- STT REST : `POST https://api.gradium.ai/api/post/speech/asr`, header `x-api-key`,
  Content-Type `audio/wav` (ou pcm), reponse NDJSON avec champs `type`/`text`.
  Source : https://docs.gradium.ai/guides/speech-to-text-rest (fetch 2026-07-05).
- TTS REST : `POST https://api.gradium.ai/api/post/speech/tts`, body JSON
  `{text, voice_id, output_format:"wav", only_audio:true}`.
  Source : https://docs.gradium.ai/guides/text-to-speech-rest (fetch 2026-07-05).
- S2S WebSocket : `wss://api.gradium.ai/api/speech/s2s`, setup
  `{model_name:"s2s-translate", stt_model_name:"stt-translate", json_config:{target_language:"en"}, voice_id}`,
  audio envoye en `{"type":"audio","audio":"<base64>"}`, fin de flux `{"type":"end_of_stream"}`.
  Source : https://docs.gradium.ai/guides/speech-to-speech (fetch 2026-07-05).
- `pip install gradium`, variable d'env `GRADIUM_API_KEY`, header `x-api-key` : confirme.
  Source : https://docs.gradium.ai/guides/installation (fetch 2026-07-05).
- Release notes actives (dernieres entrees 2026-06-10), doc a jour au 05/07/2026.
  Source : https://docs.gradium.ai/guides/release-notes
- Recuperer les vrais `voice_id` via l'endpoint Get Voices (jamais de voice_id code en dur).

---

## TL;DR jour J

1. **Tester la cle Gemini des reception** (risque n1) : depuis le 19/06/2026 une cle
   standard non restreinte est rejetee. Commande de test en section 1.
2. **Essayer la Voie 1 `environment: "desktop"`** (nouvelle depuis fin juin 2026) ;
   la Voie 2 (vision grounding, implementee et testee) reste le plan solide.
3. Modele par defaut : `gemini-3.5-flash` (GA, Computer Use integre). Fallbacks dans
   `.env.example`.
4. Nemotron/OpenRouter et Gradium : confirmes, rien a changer.
