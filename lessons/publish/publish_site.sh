#!/usr/bin/env bash
# =============================================================================
# publish_site.sh — Publish the student-facing CE6021 lesson site.
#
# Usage:
#   lessons/publish/publish_site.sh --create        # first time: make the repo
#   lessons/publish/publish_site.sh --update        # afterwards: push changes
#   lessons/publish/publish_site.sh --dry-run       # build only, push nothing
#   lessons/publish/publish_site.sh --delete        # remove the remote repo
#
# What gets published
#   ONLY the rendered site: HTML pages, style.css, site.js and figures/.
#   The lesson markdown, the figure engines, the build system and this repo's
#   git history are never copied into the public repo, so a third party cannot
#   rebuild or re-theme the course from what is published.
#
# Releasing lessons a few at a time
#   lessons/publish/released.txt lists the lesson ids to publish. Lessons that
#   are not listed get no page and no figures in the published site; they show
#   as "coming soon" on the index and are not linked from any page. Add ids to
#   that file and re-run with --update to release more.
#
# Repository visibility — READ THIS
#   The PUBLISHED SITE IS ALWAYS PUBLIC. Anyone with the URL can read it, with
#   no sign-in. Making a Pages site itself private requires GitHub Enterprise
#   Cloud and an organization-owned repository, which this script cannot do.
#
#   The --private/--public choice only affects the REPOSITORY on github.com:
#     private (default)  no fork button, not clonable, not in GitHub search.
#                        Requires GitHub Pro, Team or Enterprise; on GitHub
#                        Free a Pages repository must be public.
#     --public-repo      the repository is browsable and forkable as well.
#
#   Either way the site is world-readable, so the real protections here are
#   that only rendered output is published and that LICENSE reserves all rights.
#
# Prerequisites
#   - GitHub CLI (gh), authenticated: gh auth login
#     (--delete additionally needs the delete_repo scope)
#   - Python 3 with the lesson build dependencies installed
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
REPO_PREFIX="CE6021_AY2627"
REPO_SUFFIX="lessons"
LESSONS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CE6021_ROOT="$(cd "${LESSONS_ROOT}/.." && pwd)"
RELEASED_FILE="${LESSONS_ROOT}/publish/released.txt"
COURSE_TITLE="CE6021 Machine Vision"
OWNER_NAME="Dr Tony Scanlan, University of Limerick"

# ── Argument parsing ─────────────────────────────────────────────────────────
DO_CREATE=false
DO_UPDATE=false
DO_DELETE=false
DO_DRYRUN=false
PUBLIC_REPO=false
MESSAGE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [--create | --update | --delete | --dry-run] [options]

  --create          Create the GitHub repository, push the site, enable Pages
  --update          Rebuild the released lessons and push them to the existing repo
  --delete          Delete the remote repository
  --dry-run         Build the site into a temp directory and stop (no network)
  --public-repo     Make the repository itself public (default: private)
  --released FILE   Release manifest to use (default: publish/released.txt)
  --message TEXT    Commit message (default: a summary of the released lessons)
  --help            Show this message

First publication:
  $(basename "$0") --create

Release more lessons later (edit released.txt first):
  $(basename "$0") --update --message "Release section 2"
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --create)      DO_CREATE=true ;;
        --update)      DO_UPDATE=true ;;
        --delete)      DO_DELETE=true ;;
        --dry-run)     DO_DRYRUN=true ;;
        --public-repo) PUBLIC_REPO=true ;;
        --released)    shift; RELEASED_FILE="$1" ;;
        --message)     shift; MESSAGE="$1" ;;
        --help|-h)     usage ;;
        *) echo "❌  Unknown argument: $1"; usage ;;
    esac
    shift
done

if [[ "$DO_CREATE" == false && "$DO_UPDATE" == false && "$DO_DELETE" == false && "$DO_DRYRUN" == false ]]; then
    echo "❌  Error: one of --create, --update, --delete or --dry-run is required."
    usage
fi

if [[ "$DO_CREATE" == true && "$DO_UPDATE" == true ]]; then
    echo "❌  Error: --create and --update are mutually exclusive."
    exit 1
fi

REPO_NAME="${REPO_PREFIX}_${REPO_SUFFIX}"

# ── Dependency checks ────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python) || {
    echo "❌  Python is not installed."; exit 1; }

if [[ "$DO_DRYRUN" == false ]]; then
    command -v gh &>/dev/null || { echo "❌  'gh' (GitHub CLI) is not installed. See https://cli.github.com"; exit 1; }
    gh auth status &>/dev/null 2>&1 || { echo "❌  Not authenticated with GitHub CLI. Run: gh auth login"; exit 1; }
    GH_USER=$(gh api user --jq .login)
    GH_PLAN=$(gh api user --jq '.plan.name // "unknown"' 2>/dev/null || echo unknown)
    # GitHub Free cannot publish Pages from a private repository.
    if [[ "$PUBLIC_REPO" == false && "$GH_PLAN" == "free" ]]; then
        echo "⚠️   Your GitHub plan is '${GH_PLAN}', which cannot serve GitHub Pages from a"
        echo "    private repository. Pages needs a public repository on GitHub Free."
        echo ""
        echo "    Re-run with --public-repo to publish, or upgrade the account."
        echo "    Note that the published site is world-readable either way; a private"
        echo "    repository only hides the repository itself, not the site."
        exit 1
    fi
else
    GH_USER="(dry run)"
    GH_PLAN="(dry run)"
fi

if [[ ! -f "$RELEASED_FILE" ]]; then
    echo "❌  Release manifest not found: ${RELEASED_FILE}"
    exit 1
fi

RELEASED_IDS=$(grep -v '^[[:space:]]*#' "$RELEASED_FILE" | sed 's/#.*//' | tr -d '[:blank:]' | grep -v '^$' || true)
RELEASED_COUNT=$(printf '%s\n' "$RELEASED_IDS" | grep -c . || true)

if [[ "$RELEASED_COUNT" -eq 0 ]]; then
    echo "❌  No lessons are listed in ${RELEASED_FILE}. Nothing to publish."
    exit 1
fi

echo "=================================================="
echo "  CE6021 Lesson Site Publisher"
echo "  GitHub user : ${GH_USER}"
echo "  Repository  : ${REPO_NAME} ($( [[ "$PUBLIC_REPO" == true ]] && echo "public" || echo "private" ))$( [[ "$DO_DRYRUN" == false ]] && echo " (plan: ${GH_PLAN})" )"
echo "  Site        : PUBLIC — readable by anyone with the URL, no sign-in"
echo "  Manifest    : ${RELEASED_FILE}"
echo "  Releasing   : ${RELEASED_COUNT} lesson(s)"
printf '                %s\n' $(printf '%s ' $RELEASED_IDS)
echo "=================================================="
echo ""

# ── Delete ───────────────────────────────────────────────────────────────────
if [[ "$DO_DELETE" == true ]]; then
    if gh repo view "${GH_USER}/${REPO_NAME}" &>/dev/null 2>&1; then
        echo "▶  Deleting ${GH_USER}/${REPO_NAME} …"
        gh repo delete "${GH_USER}/${REPO_NAME}" --yes
        echo "   Deleted."
    else
        echo "   ${GH_USER}/${REPO_NAME} does not exist — nothing to delete."
    fi
    [[ "$DO_CREATE" == false ]] && exit 0
fi

# ── Step 1: Build the released subset into a temp directory ──────────────────
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT
BUILD_DIR="${WORK_DIR}/site"

echo "▶ [1/4]  Building the released lessons …"
"$PYTHON" "${LESSONS_ROOT}/build/build_site.py" \
    --out "$BUILD_DIR" --clean --released-file "$RELEASED_FILE"
echo ""

# Sanity check: one page per released lesson, plus the index.
for lid in $RELEASED_IDS; do
    if [[ ! -f "${BUILD_DIR}/${lid}.html" ]]; then
        echo "❌  Build did not produce ${lid}.html — aborting before any push."
        exit 1
    fi
done
[[ -f "${BUILD_DIR}/index.html" ]] || { echo "❌  Build did not produce index.html — aborting."; exit 1; }

# Sanity check: nothing unreleased leaked into the output.
LEAKED=$(find "$BUILD_DIR" -maxdepth 1 -name '[0-9][0-9]_[0-9][0-9].html' -printf '%f\n' \
         | sed 's/\.html$//' | sort > "${WORK_DIR}/have.txt"; \
         printf '%s\n' $RELEASED_IDS | sort > "${WORK_DIR}/want.txt"; \
         comm -23 "${WORK_DIR}/have.txt" "${WORK_DIR}/want.txt")
if [[ -n "$LEAKED" ]]; then
    echo "❌  Unreleased lesson pages found in the build: ${LEAKED}"
    exit 1
fi

# Sanity check: no dead internal links.
"$PYTHON" - "$BUILD_DIR" <<'PYCHECK'
import re, sys, pathlib
d = pathlib.Path(sys.argv[1])
have = {p.name for p in d.glob("*.html")}
bad = []
for p in sorted(d.glob("*.html")):
    for href in set(re.findall(r'href="([^":#?]+\.html)"', p.read_text(encoding="utf-8"))):
        if href not in have:
            bad.append(f"{p.name} -> {href}")
if bad:
    print("dead internal links:", *bad, sep="\n  ")
    sys.exit(1)
print(f"   link check passed across {len(have)} pages")
PYCHECK

# ── Step 2: Add the files that only the published site needs ─────────────────
echo ""
echo "▶ [2/4]  Adding publication files …"

# Discourage crawlers and mirrors. Students reach the site from Brightspace.
cat > "${BUILD_DIR}/robots.txt" <<'ROBOTS'
User-agent: *
Disallow: /
ROBOTS
echo "         → robots.txt (site excluded from search engines)"

# No open-source licence: all rights reserved, so copying is plainly an
# infringement. The Commons source images keep their own licences, which are
# printed under every figure and listed at the foot of each lesson.
cat > "${BUILD_DIR}/LICENSE" <<LICENCE
Copyright (c) $(date +%Y) ${OWNER_NAME}

All rights reserved.

The lesson text, the generated figures and the layout of this site are the
copyright of the author and are provided solely for the use of students
enrolled on ${COURSE_TITLE} at the University of Limerick.

No permission is granted to copy, redistribute, mirror, republish, adapt or
use this material, in whole or in part, for any other purpose, including
training machine learning models, without the written permission of the
author.

Photographs used in the figures are reproduced from Wikimedia Commons under
their own licences. Each is credited beneath the figure that uses it and in
the "Image credits" list at the foot of the lesson page, and those licences
continue to apply to those images.
LICENCE
echo "         → LICENSE (all rights reserved)"

# Provenance stamp: identifies which build a copy came from.
cat > "${BUILD_DIR}/build-info.txt" <<INFO
${COURSE_TITLE} — student lesson site
built    $(date -u '+%Y-%m-%d %H:%M:%S UTC')
author   ${OWNER_NAME}
lessons  $(printf '%s ' $RELEASED_IDS)
INFO
echo "         → build-info.txt"

# Tell GitHub Pages not to run the files through Jekyll.
touch "${BUILD_DIR}/.nojekyll"
echo "         → .nojekyll"

cat > "${BUILD_DIR}/README.md" <<README
# ${COURSE_TITLE} — Lessons

Student-facing lesson pages for ${COURSE_TITLE}, University of Limerick.

This repository holds **generated output only**. The lesson sources, the
Python figure engines and the build system live in a separate private
repository and are not published here.

Copyright (c) $(date +%Y) ${OWNER_NAME}. All rights reserved — see LICENSE.
README
echo "         → README.md"

PAGE_COUNT=$(find "$BUILD_DIR" -maxdepth 1 -name '*.html' | wc -l)
FIG_COUNT=$(find "${BUILD_DIR}/figures" -name '*.png' 2>/dev/null | wc -l)
SITE_SIZE=$(du -sh "$BUILD_DIR" | cut -f1)
echo ""
echo "         ${PAGE_COUNT} page(s), ${FIG_COUNT} figure(s), ${SITE_SIZE} total"

if [[ "$DO_DRYRUN" == true ]]; then
    KEEP="${CE6021_ROOT}/lessons/site_preview"
    rm -rf "$KEEP"
    cp -r "$BUILD_DIR" "$KEEP"
    echo ""
    echo "=================================================="
    echo "  ✅  Dry run complete. Nothing was pushed."
    echo "  Preview: ${KEEP}/index.html"
    echo "=================================================="
    exit 0
fi

# ── Step 3: Put the build into a clean git repo ──────────────────────────────
echo ""
echo "▶ [3/4]  Preparing the repository …"
REPO_DIR="${WORK_DIR}/repo"

REPO_EXISTS=false
gh repo view "${GH_USER}/${REPO_NAME}" &>/dev/null 2>&1 && REPO_EXISTS=true

if [[ "$DO_UPDATE" == true && "$REPO_EXISTS" == false ]]; then
    echo "❌  ${GH_USER}/${REPO_NAME} does not exist. Run with --create first."
    exit 1
fi
if [[ "$DO_CREATE" == true && "$REPO_EXISTS" == true ]]; then
    echo "❌  ${GH_USER}/${REPO_NAME} already exists. Use --update, or --delete --create."
    exit 1
fi

GIT_NAME=$(git config --global user.name 2>/dev/null || true)
GIT_EMAIL=$(git config --global user.email 2>/dev/null || true)
if [[ -z "$GIT_NAME" || "$GIT_NAME" == "null" ]]; then
    GIT_NAME=$(gh api user --jq '.name // empty' 2>/dev/null || true)
    [[ -z "$GIT_NAME" || "$GIT_NAME" == "null" ]] && GIT_NAME="$GH_USER"
fi
[[ -z "$GIT_EMAIL" || "$GIT_EMAIL" == "null" ]] && GIT_EMAIL="${GH_USER}@users.noreply.github.com"

if [[ "$DO_UPDATE" == true ]]; then
    echo "         Cloning ${GH_USER}/${REPO_NAME} …"
    git -c "credential.helper=!gh auth git-credential" \
        clone --depth 1 "https://github.com/${GH_USER}/${REPO_NAME}.git" "$REPO_DIR" --quiet
    cd "$REPO_DIR"
    # Replace the whole working tree, so un-releasing a lesson removes its page
    # and its figures rather than leaving them orphaned on the site.
    git rm -r --quiet --cached . >/dev/null 2>&1 || true
    find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
else
    mkdir -p "$REPO_DIR"
    cd "$REPO_DIR"
    git init -b main --quiet
fi

cp -r "${BUILD_DIR}/." "$REPO_DIR/"
git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"
git add -A

if [[ -z "$MESSAGE" ]]; then
    MESSAGE="Publish ${RELEASED_COUNT} lesson(s): $(printf '%s ' $RELEASED_IDS)"
fi

if git diff --cached --quiet; then
    echo "         No changes since the last publication — nothing to push."
    echo ""
    echo "✅  Site is already up to date."
    exit 0
fi

git commit --quiet -m "$MESSAGE"
echo "         Committed: ${MESSAGE}"

# ── Step 4: Create or update the remote, and enable Pages ────────────────────
echo ""
echo "▶ [4/4]  Pushing to GitHub …"

if [[ "$DO_CREATE" == true ]]; then
    VISIBILITY_FLAG="--private"
    [[ "$PUBLIC_REPO" == true ]] && VISIBILITY_FLAG="--public"
    gh repo create "${REPO_NAME}" ${VISIBILITY_FLAG} \
        --description "${COURSE_TITLE} — student lesson pages (generated output only)"
    git remote add origin "https://github.com/${GH_USER}/${REPO_NAME}.git"
fi

git -c "credential.helper=!gh auth git-credential" push --quiet origin main

if [[ "$DO_CREATE" == true ]]; then
    echo "         Enabling GitHub Pages on main …"
    gh api --method POST "repos/${GH_USER}/${REPO_NAME}/pages" \
        --field "source[branch]=main" --field "source[path]=/" >/dev/null 2>&1 \
        || echo "         ⚠  Could not enable Pages automatically."
    echo "            If Pages did not turn on, check Settings → Pages. Serving a"
    echo "            private repository through Pages needs GitHub Pro, Team or"
    echo "            Enterprise; on a free account re-run with --public-repo."
fi

PAGES_URL=$(gh api "repos/${GH_USER}/${REPO_NAME}/pages" --jq .html_url 2>/dev/null || true)

echo ""
echo "=================================================="
echo "  ✅  Published successfully!"
echo "  Repository : https://github.com/${GH_USER}/${REPO_NAME}"
[[ -n "$PAGES_URL" ]] && echo "  Site       : ${PAGES_URL}"
echo "  Lessons    : ${RELEASED_COUNT} released"
echo ""
echo "  The site is publicly readable by anyone with the URL. Link it from"
echo "  Brightspace. To release more lessons, uncomment their ids in"
echo "  $(basename "$RELEASED_FILE") and run --update."
echo "=================================================="
