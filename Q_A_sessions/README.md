# Q_A_sessions/

Slide decks (PowerPoint, `.pptx`) for the CE5021 Q&A sessions that support the
assignments. The layout follows `lessons/`: content files, figure engines with the
same contract, and a build script.

| Path | Contents |
|---|---|
| `content/` | One YAML file per session: deck title, engines, and the slides (layout, title, figure, cards, code, speaker notes) |
| `engines/` | Figure engines, one package per session. Each module has `FIGURES` and `build(ctx) -> list[Figure]`, as in `lessons/engines`, and imports the shared helpers from `lessons/engines/_common.py` |
| `build/build_decks.py` | Runs the engines and writes the decks with `python-pptx` |
| `build/theme.py` | Colours, fonts and sizes, taken from the lesson pages and figure palette |
| `output/` | Generated figures and decks (gitignored; rebuild with the script) |

## Build

```bash
python Q_A_sessions/build/build_decks.py              # every deck
python Q_A_sessions/build/build_decks.py --only Q_A_1
```

The deck is written to `Q_A_sessions/output/<output>.pptx`. The figure captions,
which state the measured values, are added to the speaker notes of each slide.

## Sessions

| Deck | Assignment | Topic | Engine |
|---|---|---|---|
| `Q_A_1` | Assignment 1 (regularisation and optimisation, CIFAR-10) | Learning-rate policies: fixed rates and when to reduce them, warm-up, cyclical rates (Smith), `torch.optim` schedulers stepped per epoch or per batch | `engines/qa1_lr` |

## Rules

- **No assignment solutions.** Examples are built on toy problems (Q&A 1: a
  three-arm spiral classified by a small MLP), never on the assignment network or
  its settings. The private assignments repository is only used to match the
  structure of the training loop and the terminology.
- **Measured figures.** As in the lessons, every curve is a real run, and the claims
  in captions and notes are checked with assertions in the engine.
- **Prose.** Plain lecturer's register, as in the lessons (see
  `lessons/COURSE_PLAN.md`, section 11).
