# Docs

Sphinx sources for the YTM documentation, published with GitHub Pages from the `gh-pages` branch.

Publishing is manual: docs are built locally, committed to the local `gh-pages` branch by [ghp-import](https://github.com/c-w/ghp-import).

## Layout

| Path | Purpose |
|---|---|
| `conf.py` | Sphinx config. `version` (major.minor) comes from the installed `ytm` package. |
| `index.md`, `examples.md`, `api/` | Doc pages. |
| `_static/switcher.json` | Version list for the version dropdown. |
| `_pages_root/` | Files for the root of `gh-pages` (`index.html` redirects to `stable/`). |
| `publish.py` | Init and publish logic used by the pixi tasks below. |

`gh-pages` branch:

```
.nojekyll
index.html      redirects to stable/
stable/         latest release
1.0/            one folder per major.minor release
```

Every version reads the switcher from `stable/_static/switcher.json`, so older versions also list newer releases.

## Tasks

| Command | What it does |
|---|---|
| `pixi run -e docs docs` | Build into `docs/_build/html` for local preview. |
| `pixi run -e docs docs-pages-init` | One time: create `gh-pages` with the root files. Refuses if `gh-pages` exists locally or on `origin`. |
| `pixi run -e docs docs-publish` | Full rebuild, then commit it to `<major.minor>/` and `stable/` on the local `gh-pages`. |

`docs-publish` refuses unless:

- `gh-pages` exists and contains `origin/gh-pages` (run `git fetch` first),
- the current branch is `main`,
- the working tree is clean.

Nothing is pushed by these tasks.

## Release

1. Bump `version` in `pyproject.toml`.
2. For a new major.minor: add an entry to `_static/switcher.json`, move `"preferred": true` and the `(stable)` label to it. Patch releases reuse the major.minor folder, no switcher change.
3. Commit, merge to `main`, check out `main`.
4. `git fetch origin`
5. `pixi run -e docs docs-publish`
6. Inspect: `git log -3 --stat gh-pages`
7. `git push origin gh-pages`

## One-time setup

1. `pixi run -e docs docs-pages-init`
2. `git push origin gh-pages`
