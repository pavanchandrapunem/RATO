#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash scripts/init_git.sh YOUR_GITHUB_USERNAME' >&2
  exit 1
fi
if [ -d .git ]; then
  echo 'Repository already initialized; refusing to overwrite origin.' >&2
  exit 1
fi
git init -b main
git add .
git commit -m 'Initial RATO independent reproduction'
git remote add origin "git@github.com:$1/RATO.git"
echo "Remote configured; first create an empty GitHub repository named RATO, then run: git push -u origin main"
