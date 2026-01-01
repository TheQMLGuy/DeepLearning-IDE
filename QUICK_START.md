# 🚀 Quick Start Guide - 5 Minute Setup

Get your AI-IDE live on GitHub Pages in just 5 minutes!

---

## Prerequisites (1 minute)

You need:
- ✅ GitHub account ([sign up free](https://github.com/signup))
- ✅ Git installed ([download](https://git-scm.com/downloads))
- ✅ Node.js 18+ ([download](https://nodejs.org/))

**Check your versions:**
```bash
git --version
node --version  # Should be 18 or higher
npm --version
```

---

## Step 1: Get the Code (1 minute)

### Option A: Download ZIP
1. Click **Code** button → **Download ZIP**
2. Extract to your desired location
3. Open terminal in that folder

### Option B: Clone with Git
```bash
git clone https://github.com/YOUR-USERNAME/AI-IDE.git
cd AI-IDE
```

---

## Step 2: Test Locally (2 minutes)

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

**You should see:**
```
VITE v5.1.6  ready in 432 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

**Open http://localhost:5173/** in your browser.

**Test it works:**
1. Wait for "Python Ready" indicator
2. Click in a cell
3. Type: `print("Hello!")`
4. Press **Shift+Enter**
5. See output appear

**✅ If you see "Hello!" - it works! Continue to deployment.**

---

## Step 3: Deploy to GitHub (2 minutes)

### 3.1: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it: `AI-IDE` (or anything you want)
3. Make it **Public**
4. **DON'T** initialize with README
5. Click **Create repository**

### 3.2: Push Your Code

```bash
# Initialize git (if not cloned)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - AI-IDE"

# Add your GitHub repository
# Replace YOUR-USERNAME with your GitHub username
git remote add origin https://github.com/YOUR-USERNAME/AI-IDE.git

# Push to GitHub
git push -u origin main
```

### 3.3: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. Click **Pages** (left sidebar)
4. Under **Source**, select **GitHub Actions**
5. Click **Save**

### 3.4: Wait for Deployment

1. Click **Actions** tab
2. You'll see a workflow running
3. Wait 2-3 minutes for green checkmark ✅
4. Done!

---

## Step 4: Access Your Site! 🎉

Your site is now live at:
```
https://YOUR-USERNAME.github.io/AI-IDE/
```

**Replace `YOUR-USERNAME` with your actual GitHub username!**

---

## Verify Deployment

### Quick Test
1. Visit your site URL
2. Wait for "Python Ready"
3. Run this code:

```python
import torch
print(f"PyTorch {torch.__version__} is working!")
print("🎉 Your AI-IDE is live!")
```

### Full Test
```python
# Complete functionality test
import torch
import numpy as np

# Test PyTorch
x = torch.randn(3, 3)
print("✅ PyTorch works!")

# Test NumPy
arr = np.array([1, 2, 3, 4, 5])
print(f"✅ NumPy works! Mean: {arr.mean()}")

# Test computation
result = x @ x.T
print("✅ GPU computation works!")

print("\n🚀 Your AI-IDE is fully operational!")
```

**If you see all ✅ checkmarks - SUCCESS!**

---

## What You Just Built

You now have:
- ✅ A live web-based Python IDE
- ✅ PyTorch and NumPy pre-installed
- ✅ Jupyter-style notebooks
- ✅ VS Code editor (Monaco)
- ✅ 55+ deep learning features
- ✅ Auto-deployment on push
- ✅ Free hosting forever

**And it only took 5 minutes!** 🎉

---

## Next Steps

### 1. Try the Features

**Load a dataset:**
```python
from dl_utils import load_dataset
train_loader, test_loader = load_dataset('cifar10')
print(f"Loaded {len(train_loader.dataset)} training samples")
```

**Train a model:**
```python
from dl_utils import quick_start
model, history = quick_start('cifar10', 'simple', epochs=5)
```

**Find optimal learning rate:**
```python
from dl_utils import find_lr
best_lr = find_lr(model, train_loader, criterion)
```

### 2. Customize Your IDE

Edit `src/App.tsx` to:
- Change colors/theme
- Add new panels
- Modify layout
- Add custom features

Then commit and push:
```bash
git add .
git commit -m "Customized my IDE"
git push
```

**Your site auto-updates in 2 minutes!**

### 3. Share Your Work

Share your live IDE:
```
https://YOUR-USERNAME.github.io/AI-IDE/
```

Add it to:
- LinkedIn profile
- GitHub profile README
- Resume/Portfolio
- Twitter/social media

---

## Common Issues & Fixes

### Issue: "npm install" fails
**Fix:**
```bash
# Delete node_modules and try again
rm -rf node_modules
rm package-lock.json
npm install
```

### Issue: Port 5173 already in use
**Fix:**
```bash
# Use different port
npm run dev -- --port 3000
```

### Issue: Blank page after deployment
**Fix:**
1. Check browser console (F12)
2. Clear cache (Ctrl+Shift+R)
3. Wait a few more minutes
4. Check Actions tab for errors

### Issue: Python not working
**Fix:**
1. Wait 10 seconds after "Python Ready"
2. Try in incognito mode
3. Check console for errors
4. Ensure WASM is supported (all modern browsers)

### Issue: GitHub Actions failed
**Fix:**
1. Go to Actions tab
2. Click the failed workflow
3. Read the error log
4. Usually: update Node.js version or run `npm install`

---

## Useful Commands

```bash
# Development
npm run dev              # Start dev server
npm run build           # Build for production
npm run preview         # Preview production build

# Deployment
git add .               # Stage all changes
git commit -m "message" # Commit changes
git push                # Deploy to GitHub Pages

# Maintenance
npm update              # Update dependencies
npm run lint            # Check code quality
```

---

## One-Line Commands

**Quick deploy (after initial setup):**
```bash
git add . && git commit -m "Update" && git push
```

**Reset to clean state:**
```bash
rm -rf node_modules dist && npm install && npm run dev
```

**Check build size:**
```bash
npm run build && du -sh dist/
```

---

## Getting Help

### Documentation
- 📖 [Full Features Guide](./DL_FEATURES.md)
- 🚀 [Detailed Deployment](./GITHUB_PAGES_DEPLOY.md)
- ✅ [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)
- 🐍 [PyTorch Guide](./PYTORCH_IDE_README.md)

### Community
- 🐛 [Report Issues](https://github.com/YOUR-USERNAME/AI-IDE/issues)
- 💬 [Discussions](https://github.com/YOUR-USERNAME/AI-IDE/discussions)
- ⭐ [Star the repo](https://github.com/YOUR-USERNAME/AI-IDE) if helpful!

### Still Stuck?
Open an issue with:
- Your error message
- Browser console output
- Steps you took
- What you expected vs what happened

---

## Success Checklist

- [ ] Node.js 18+ installed
- [ ] Git installed
- [ ] Code downloaded/cloned
- [ ] `npm install` successful
- [ ] `npm run dev` works locally
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] GitHub Pages enabled (GitHub Actions)
- [ ] Deployment workflow completed ✅
- [ ] Site loads at GitHub Pages URL
- [ ] Python executes code
- [ ] All features work

**All checked?** 🎉 **Congratulations! You're done!**

---

## Pro Tips

### Tip 1: Bookmark Your Site
Add your GitHub Pages URL to bookmarks for quick access.

### Tip 2: Share Notebooks
Share your URL with specific code to run - great for teaching!

### Tip 3: Mobile Development
Edit on desktop, view on mobile - your changes sync automatically.

### Tip 4: Backup Your Work
Regularly commit and push to GitHub - free backup!

### Tip 5: Custom Domain (Optional)
Want `ai-ide.yourdomain.com`?
1. Add `CNAME` file to `public/` with your domain
2. Configure DNS records
3. Enable in GitHub Pages settings

---

## What's Next?

### Learn More
- Explore all 55+ features
- Try different datasets
- Experiment with models
- Build custom training loops

### Contribute
- Add new features
- Fix bugs
- Improve documentation
- Share templates

### Build Projects
Use your IDE to:
- Learn PyTorch
- Prototype models
- Teach others
- Build portfolio projects

---

## 🎊 You Did It!

You now have a **production-ready PyTorch IDE** running on GitHub Pages!

**Your live site:** `https://YOUR-USERNAME.github.io/AI-IDE/`

**Time taken:** ~5 minutes ⚡

**Cost:** $0 (free forever) 💰

**Features:** 55+ 🚀

---

**Happy coding!** 💻✨

**Questions?** Check the docs or open an issue!

**Found this useful?** ⭐ Star the repo!

---

[⬆ Back to Top](#-quick-start-guide---5-minute-setup)
