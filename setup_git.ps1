# Git Repository Setup Script for Windows PowerShell
# This script prepares the repository for GitHub upload

Write-Host "🚀 Setting up Git repository for Iris Diabetes Detection..." -ForegroundColor Green

# Initialize git if not already done
if (-not (Test-Path ".git")) {
    Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow
    git init
}

# Add remote origin (replace with your actual repository URL)
$REPO_URL = "https://github.com/vignshh7/Iris-Diabetes-Detection.git"
Write-Host "🔗 Setting remote origin to: $REPO_URL" -ForegroundColor Cyan

# Remove existing origin if it exists and add new one
try { git remote remove origin 2>$null } catch { }
git remote add origin $REPO_URL

# Create all necessary directories
Write-Host "📁 Creating directory structure..." -ForegroundColor Yellow
python config.py

# Stage all files according to .gitignore
Write-Host "📝 Staging files for commit..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
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

Write-Host "🌟 Repository prepared successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Review the staged files: git status"
Write-Host "2. Force push to replace repository: git push -f origin main"
Write-Host "3. Verify .gitignore is working correctly"
Write-Host ""
Write-Host "🔍 Files that will be uploaded:" -ForegroundColor Yellow
git ls-files | Where-Object { $_ -match "\.(py|md|txt|json)$" } | Select-Object -First 20
Write-Host "... and more (see 'git ls-files' for complete list)"