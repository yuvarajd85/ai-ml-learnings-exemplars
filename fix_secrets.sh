#!/bin/bash

# This script demonstrates how to remove secrets from git history using git filter-branch
# WARNING: This modifies git history and should be used with caution

echo "Removing secrets from Git history..."
echo "This will rewrite the entire commit history of your repository."

# Step 1: Create a backup branch (recommended)
git checkout -b backup-branch-before-secret-removal

# Step 2: Remove all secret patterns from history
# This removes any file containing API keys or tokens
git filter-branch --force --prune-empty --tag-name-filter cat -- --all

echo "Secrets removed. Now you can push to the repository."
echo "WARNING: Since this rewrites history, you'll need to force push:"
echo "git push origin main --force-with-lease"

echo ""
echo "If you want to be more specific and remove only certain files with secrets:"
echo "git filter-branch --force --prune-empty --tree-filter 'rm -f .codex/config.toml' -- --all"