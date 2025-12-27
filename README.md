# AI IDE - Web-Based PyTorch Development Environment

A fully browser-based AI/ML development environment with PyTorch neural network building, interactive DataFrame viewer, and R-style statistical analysis.

**[Live Demo](https://your-username.github.io/AI-IDE/)**

## Features

- 🔥 **PyTorch Architecture Builder** - Visual layer palette to build neural networks
- 🐼 **Interactive Pandas Viewer** - Editable DataFrame with code sync
- 📊 **R Statistical Analysis** - t-test, ANOVA, regression, and more
- 📁 **Dataset Browser** - Kaggle, PyTorch, HuggingFace datasets
- 🐍 **In-Browser Python** - Real execution via Pyodide (WebAssembly)

## Quick Start

```bash
# Install dependencies
npm install

# Development
npm run dev

# Build for production
npm run build

# Preview build
npm run preview
```

## Deploy to GitHub Pages

### Option 1: GitHub Actions (Recommended)

1. Push to `main` branch
2. Go to Settings → Pages → Source: GitHub Actions
3. Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

### Option 2: Manual

```bash
npm run build
# Upload contents of dist/ to gh-pages branch
```

## Tech Stack

- React + TypeScript
- Vite
- Monaco Editor
- Pyodide (Python in WebAssembly)

## License

MIT
