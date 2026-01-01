#!/bin/bash

# AI IDE Improvement Migration Script
# This script safely migrates your existing AI IDE to the improved version

set -e  # Exit on error

echo "🚀 AI IDE Improvement Migration Script"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Are you in the AI-IDE directory?"
    exit 1
fi

echo "📋 Step 1: Creating backups..."
mkdir -p .backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=".backups/$(date +%Y%m%d_%H%M%S)"

# Backup existing files
cp src/webMocks.ts "$BACKUP_DIR/webMocks.ts" 2>/dev/null || echo "  - webMocks.ts not found, skipping"
cp src/components/notebook/Notebook.tsx "$BACKUP_DIR/Notebook.tsx" 2>/dev/null || echo "  - Notebook.tsx not found, skipping"
cp src/components/notebook/NotebookCell.tsx "$BACKUP_DIR/NotebookCell.tsx" 2>/dev/null || echo "  - NotebookCell.tsx not found, skipping"
cp src/components/notebook/NotebookCell.css "$BACKUP_DIR/NotebookCell.css" 2>/dev/null || echo "  - NotebookCell.css not found, skipping"

echo "✅ Backups created in $BACKUP_DIR"
echo ""

echo "📦 Step 2: Migrating improved files..."

# Function to safely move improved files
migrate_file() {
    local src=$1
    local dest=$2
    
    if [ -f "$src" ]; then
        cp "$src" "$dest"
        echo "  ✓ Migrated: $dest"
    else
        echo "  ⚠️  Warning: $src not found"
    fi
}

# Migrate files
migrate_file "src/webMocks.improved.ts" "src/webMocks.ts"
migrate_file "src/components/notebook/Notebook.improved.tsx" "src/components/notebook/Notebook.tsx"
migrate_file "src/components/notebook/NotebookCell.improved.tsx" "src/components/notebook/NotebookCell.tsx"
migrate_file "src/components/notebook/NotebookCell.improved.css" "src/components/notebook/NotebookCell.css"

echo ""
echo "🔧 Step 3: Creating ErrorBoundary component..."

# ErrorBoundary should already exist, but verify
if [ -f "src/components/common/ErrorBoundary.tsx" ]; then
    echo "  ✓ ErrorBoundary.tsx already exists"
else
    echo "  ⚠️  Warning: ErrorBoundary.tsx not found. Please copy it manually."
fi

echo ""
echo "📝 Step 4: Updating App.tsx..."

# Check if ErrorBoundary is imported in App.tsx
if grep -q "ErrorBoundary" "src/App.tsx"; then
    echo "  ✓ ErrorBoundary already imported in App.tsx"
else
    echo "  ⚠️  Action required: Add ErrorBoundary to App.tsx"
    echo ""
    echo "  Add this import:"
    echo "    import ErrorBoundary from './components/common/ErrorBoundary';"
    echo ""
    echo "  Wrap your app in ErrorBoundary:"
    echo "    <ErrorBoundary>"
    echo "      <div className=\"ide-container\">"
    echo "        {/* existing code */}"
    echo "      </div>"
    echo "    </ErrorBoundary>"
    echo ""
fi

echo ""
echo "🧪 Step 5: Running type checks..."

if command -v npm &> /dev/null; then
    echo "  Running TypeScript compiler..."
    npm run build 2>&1 | head -n 20 || echo "  ⚠️  Build warnings (check full output)"
else
    echo "  ⚠️  npm not found, skipping type check"
fi

echo ""
echo "✅ Migration Complete!"
echo ""
echo "📋 Next Steps:"
echo "  1. Review the changes in your IDE"
echo "  2. Test the application: npm run dev"
echo "  3. Check IMPROVEMENTS.md for detailed documentation"
echo "  4. If issues occur, restore from $BACKUP_DIR"
echo ""
echo "🎯 Key Improvements:"
echo "  ✓ Memory leak fixes"
echo "  ✓ Execution queue (prevents race conditions)"
echo "  ✓ Keyboard shortcuts (Shift+Enter, Ctrl+S, etc.)"
echo "  ✓ Error boundaries (crash prevention)"
echo "  ✓ Performance optimizations (React.memo)"
echo "  ✓ Auto-save functionality"
echo "  ✓ Better error handling"
echo ""
echo "Need help? Check IMPROVEMENTS.md for details!"
echo ""
