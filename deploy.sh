#!/bin/bash

# AI-IDE GitHub Pages Deployment Script
# Automatically builds and prepares for GitHub Pages deployment

set -e  # Exit on error

echo "🚀 AI-IDE GitHub Pages Deployment"
echo "=================================="
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo "✅ Dependencies installed"
    echo ""
fi

# Clean previous build
if [ -d "dist" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf dist
    echo "✅ Cleaned"
    echo ""
fi

# Build the project
echo "🔨 Building project..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
else
    echo "❌ Build failed!"
    exit 1
fi

# Verify build
if [ -f "dist/index.html" ]; then
    echo "✅ Build verified - index.html exists"
else
    echo "❌ Build verification failed - index.html not found"
    exit 1
fi

echo ""
echo "📊 Build Statistics:"
echo "-------------------"
du -sh dist/
echo ""
echo "📁 Build contents:"
ls -lh dist/
echo ""

echo "✅ Build complete and ready for deployment!"
echo ""
echo "📋 Next steps:"
echo "  1. Commit and push to GitHub:"
echo "     git add ."
echo "     git commit -m 'Deploy to GitHub Pages'"
echo "     git push origin main"
echo ""
echo "  2. Wait 2-3 minutes for GitHub Actions to deploy"
echo ""
echo "  3. Visit your site at:"
echo "     https://YOUR-USERNAME.github.io/AI-IDE/"
echo ""
echo "🎉 Happy deploying!"
