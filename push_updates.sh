#!/bin/bash
# Smart Automation for TBFusionAI Git Workflow

echo "🔍 Checking for modified files..."

# Get list of modified and untracked files
MODIFIED_FILES=$(git status --porcelain | awk '{print $2}')

if [ -z "$MODIFIED_FILES" ]; then
    echo "✨ No changes detected. Nothing to commit."
    exit 0
fi

echo "📦 Files to be staged:"
echo "$MODIFIED_FILES"
echo "------------------------------"

# Stage all changes
git add .

# Prompt for a commit message
echo "📝 Enter commit message (or press Enter for default 'fix: automated update'): "
read USER_MESSAGE

if [ -z "$USER_MESSAGE" ]; then
    COMMIT_MSG="fix: automated update of modified files"
else
    COMMIT_MSG="$USER_MESSAGE"
fi

echo "💾 Committing with message: '$COMMIT_MSG'..."
git commit -m "$COMMIT_MSG"

echo "📤 Pushing to main..."
git push origin main

echo "✅ Update complete!"

# #!/bin/bash
# # Automation for TBFusionAI Git Workflow

# echo "🚀 Staging modified files..."
# git add src/api/routes.py src/models/predictor.py tests/conftest.py

# echo "💾 Committing fixes (Pointer Reset & Test Fixtures)..."
# git commit -m "fix: resolve BytesIO pointer bug and fix missing pytest fixtures"

# echo "📤 Pushing to main..."
# git push origin main

# echo "✅ Done! Monitor your Google Cloud Build for deployment status."