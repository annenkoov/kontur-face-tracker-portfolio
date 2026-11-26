# Kontur — facial expression tracker

Windows desktop application built with Python, Qt Quick/QML and computer-vision models. It compares facial-expression signals with a short personal baseline, shows changes over a session, and keeps a local history. The experimental score is **not a medical stress measurement** and has not been validated for diagnosis or clinical decisions.

## Versions in this repository

- `versions/01-qtquick-main` — main Qt Quick implementation with the latest camera, analysis, and eye-state fixes.
- `versions/02-qtquick-dashboard-fork` — separate Qt Quick UI/analysis iteration with local labeling, validation and camera improvements.
- `versions/03-kirigami-fork` — separate KDE Kirigami-oriented fork. It remains an independent implementation track.

Each folder is a self-contained source snapshot. The original worktrees had separate Git histories; these folders preserve the three reviewed code states for comparison, not the full chat-by-chat revision history.

## Run on Windows

Choose one version folder, install Python 3.13 x64, then run `setup.bat` followed by `run.bat`. The app may download model files on first launch. Camera availability and inference speed depend on the device; CPU inference can be slow.

See each version's source and `requirements.txt`. Third-party model licenses and restrictions are documented in `models/README.md` and `THIRD_PARTY_NOTICES.txt`.

## Project story

See [`APPLICATION_ANSWER_RU.md`](APPLICATION_ANSWER_RU.md) for a concise first-person project description and an example of a problem fixed during development.