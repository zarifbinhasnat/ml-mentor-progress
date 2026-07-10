# Repo automation notes — read before running the daily lesson routine

This repo is driven by a scheduled agent session that generates one lesson
per day from `progress.json`. It has no human reviewer in the loop, so the
agent must close its own loop every run. Two rules prevent duplicate/orphaned
work:

## 1. Before reading `progress.json`, check for unmerged prior runs

Each scheduled run starts on a fresh branch and only reads `progress.json`
from its local clone of `main`. If a previous run's branch/PR never got
merged, `main`'s `progress.json` is stale, and a new run will silently
recompute the *same* "next concept" the unmerged run already covered —
producing duplicate `LESSON_*_DAY_N.md` files across multiple branches.

Before STEP 2 (determine today's concept), list open pull requests against
`main`. If any exist:
- If it's a clean, complete lesson for the very next concept, merge it first,
  then re-`git pull origin main` before reading `progress.json`.
- If it's a stale/duplicate/conflicting leftover (e.g. superseded by another
  run that already merged), close it rather than merging it, with a short
  comment explaining why.

Only proceed to generate a new lesson once `main` has no dangling open PRs
left over from a previous day.

## 2. After pushing, merge back into `main` before finishing

STEP 7 of the lesson routine commits and pushes to a feature branch, but
that alone is not enough — a branch nobody merges is invisible to the next
day's run. Every run must, as its last step:

1. Open a PR from its feature branch into `main`.
2. Merge that PR immediately (no human review is expected on this repo).
3. Delete the feature branch afterward to keep the branch list clean.

Skipping this step is the single most common cause of duplicate lessons:
without it, `progress.json` on `main` never advances, so every subsequent
day's run starts from the same stale state and redoes the same "next"
concept on yet another new branch.
