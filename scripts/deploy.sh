#!/bin/bash
# Simple deployment script for Prepzo Bot

# Update version.py with current date
today=$(date +"%Y-%m-%d")
sed -i "s/BUILD_DATE = \".*\"/BUILD_DATE = \"$today\"/" version.py

# Get the current git commit
git_commit=$(git rev-parse HEAD)
echo "Deploying commit: $git_commit"

# Check if there are uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "Warning: You have uncommitted changes. Commit before deploying."
    exit 1
fi

# Update version if specified
if [[ ! -z "$1" ]]; then
    echo "Updating version to $1"
    sed -i "s/VERSION = \".*\"/VERSION = \"$1\"/" version.py
    git add version.py
    git commit -m "Update version to $1"
fi

# Push to GitHub to trigger GitHub Actions
echo "Pushing to GitHub..."
git push origin main

echo "Deployment initiated. Check GitHub Actions for progress."
echo "https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions" 