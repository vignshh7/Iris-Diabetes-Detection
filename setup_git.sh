#!/bin/bash

# Git Repository Setup Script for Iris Diabetes Detection Project
# This script prepares the repository for GitHub upload

echo "🚀 Setting up Git repository for Iris Diabetes Detection..."

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
fi

# Add remote origin (replace with your actual repository URL)
REPO_URL="https://github.com/vignshh7/Iris-Diabetes-Detection.git"
echo "🔗 Setting remote origin to: $REPO_URL"

# Remove existing origin if it exists and add new one
git remote remove origin 2>/dev/null || true
git remote add origin $REPO_URL

# Create all necessary directories
echo "📁 Creating directory structure..."
python config.py

# Stage all files according to .gitignore
echo "📝 Staging files for commit..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "🎯 Complete project restructure with organized folders

✅ Features:
- Organized project structure (src/, models/, docs/, results/)
- Updated Python scripts with config-based paths
- Comprehensive .gitignore for data/temp files
- Sample visualizations with proper layout
- Results table as image format
- Non-overlapping patient IDs in samples

🏗️ Structure:
- src/ - All Python source code
- models/ - Trained model files (.pth)
- docs/ - Documentation and summaries
- results/ - Performance results and metrics
- config.py - Centralized configuration

🚫 Excluded (via .gitignore):
- dataset/ - Training and test images
- test_results/ - Generated results
- performance_analysis/ - Analysis outputs
- temp/ - Temporary files
- __pycache__/ - Python cache files

🎯 Ready for: Research, Development, Deployment
📊 Performance: 92.2% accuracy with 5-fold cross-validation"

echo "🌟 Repository prepared successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Review the staged files: git status"
echo "2. Push to GitHub: git push -u origin main"
echo "3. Verify .gitignore is working correctly"
echo ""
echo "🔍 Files that will be uploaded:"
git ls-files | grep -E "\.(py|md|txt|json|csv)$" | head -20
echo "... and more (see 'git ls-files' for complete list)"