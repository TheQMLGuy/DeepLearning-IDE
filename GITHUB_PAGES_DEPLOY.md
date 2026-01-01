# 🚀 GitHub Pages Deployment Guide

This guide will help you deploy your AI-IDE to GitHub Pages in less than 5 minutes.

---

## ✅ Prerequisites

1. GitHub account
2. Git installed locally
3. Node.js 18+ installed

---

## 📋 Deployment Methods

### Method 1: Automatic Deployment (Recommended) ⭐

This method uses GitHub Actions to automatically build and deploy whenever you push to the main branch.

#### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - AI-IDE ready for deployment"

# Add remote (replace with your repository URL)
git remote add origin https://github.com/YOUR-USERNAME/AI-IDE.git

# Push to main branch
git push -u origin main
```

#### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select **GitHub Actions**
4. Wait 2-3 minutes for the deployment to complete

#### Step 3: Access Your Site

Your site will be available at:
```
https://YOUR-USERNAME.github.io/AI-IDE/
```

**That's it!** 🎉 Every time you push to main, it will automatically redeploy.

---

### Method 2: Manual Deployment

If you prefer to deploy manually:

#### Step 1: Build the Project

```bash
# Install dependencies
npm install

# Build for production
npm run build
```

#### Step 2: Deploy

```bash
# Install gh-pages package
npm install -D gh-pages

# Add deploy script to package.json
# "deploy": "gh-pages -d dist"

# Deploy
npm run deploy
```

#### Step 3: Configure GitHub Pages

1. Go to **Settings** → **Pages**
2. Under **Source**, select **Deploy from a branch**
3. Select **gh-pages** branch
4. Click **Save**

---

## 🔧 Configuration Details

### Files That Make It Work

1. **`.github/workflows/deploy.yml`** - GitHub Actions workflow
   - Automatically builds and deploys on push to main
   - Uses Node.js 20
   - Caches dependencies for faster builds

2. **`vite.config.ts`** - Vite configuration
   - `base: './'` - Uses relative paths (works for any subdirectory)
   - Optimized build settings
   - Code splitting for faster load times

3. **`public/.nojekyll`** - Prevents Jekyll processing
   - GitHub Pages uses Jekyll by default
   - This file disables it for better compatibility

### Custom Domain (Optional)

To use a custom domain like `ai-ide.yourdomain.com`:

1. Create a file `public/CNAME` with your domain:
   ```
   ai-ide.yourdomain.com
   ```

2. Configure DNS:
   - Add a CNAME record pointing to `YOUR-USERNAME.github.io`
   
3. In GitHub Settings → Pages, enter your custom domain

---

## 🐛 Troubleshooting

### Issue: Page shows blank screen

**Solution:** Check browser console for errors. Common causes:
- CORS issues with external resources
- Wrong base path in vite.config.ts

**Fix:** Ensure `vite.config.ts` has `base: './'`

### Issue: Monaco Editor not loading

**Solution:** Monaco Editor needs special handling:
```typescript
// Already configured in vite.config.ts
rollupOptions: {
  output: {
    manualChunks: {
      'monaco': ['monaco-editor', '@monaco-editor/react']
    }
  }
}
```

### Issue: Deployment fails in GitHub Actions

**Solution:** Check the Actions tab for error logs:
1. Go to **Actions** tab
2. Click on the failed workflow
3. Expand the failed step to see error details

Common fixes:
- Ensure `package.json` has correct scripts
- Check if all dependencies are in package.json
- Verify Node.js version (should be 18+)

### Issue: 404 errors for assets

**Solution:** Verify the base path:
```typescript
// vite.config.ts
export default defineConfig({
  base: './', // Relative paths work everywhere
})
```

### Issue: Python/Pyodide not working

**Solution:** Pyodide loads from CDN, which may be blocked:
- Check browser console for CORS errors
- Ensure network connectivity
- CDN URL: `https://cdn.jsdelivr.net/pyodide/v0.24.1/full/`

---

## 🔍 Verify Deployment

After deployment, test these features:

1. ✅ **Page loads** - No blank screen
2. ✅ **Python execution** - Run a test cell
3. ✅ **Code editor** - Monaco editor appears
4. ✅ **Panels** - All side panels open/close correctly
5. ✅ **Network** - Check browser DevTools → Network tab for failed requests

---

## 📊 Build Optimization

Your deployment is already optimized with:

### 1. Code Splitting
```typescript
manualChunks: {
  'monaco': ['monaco-editor', '@monaco-editor/react'],
  'vendor': ['react', 'react-dom']
}
```

### 2. Asset Optimization
- Images compressed
- CSS minified
- JS bundled and minified

### 3. Caching
- GitHub Actions caches npm dependencies
- Browser caches static assets

### Build Size Breakdown:
```
dist/
├── assets/
│   ├── monaco-*.js      (~2MB - code editor)
│   ├── vendor-*.js      (~500KB - React)
│   ├── index-*.js       (~300KB - app code)
│   └── index-*.css      (~50KB - styles)
└── index.html           (~5KB)

Total: ~3MB (loads in 2-3 seconds on 4G)
```

---

## 🚀 Performance Tips

### 1. Enable Compression (Already Done)
Vite automatically minifies and compresses all assets.

### 2. Lazy Loading (Already Implemented)
Monaco Editor is code-split and loads only when needed.

### 3. CDN Usage (Already Configured)
- Pyodide loads from jsdelivr CDN
- Fonts load from system or Google Fonts
- External libraries from CDN

### 4. Caching Strategy
```html
<!-- Already in index.html -->
<meta http-equiv="Cache-Control" content="max-age=31536000">
```

---

## 🔄 Update Your Deployment

### Automatic Updates
Just push to main branch:
```bash
git add .
git commit -m "Update: Added new feature"
git push
```

GitHub Actions will automatically rebuild and redeploy (takes 2-3 minutes).

### Manual Updates
```bash
npm run build
npm run deploy
```

---

## 📱 Mobile Compatibility

The IDE is mobile-responsive with:
- Touch-friendly controls
- Responsive layout
- Mobile-optimized code editor
- Swipe gestures for panels

Test on mobile by visiting:
```
https://YOUR-USERNAME.github.io/AI-IDE/
```

---

## 🌐 Cross-Browser Support

Tested and working on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

**Note:** Pyodide requires WebAssembly support (all modern browsers).

---

## 📈 Analytics (Optional)

To add Google Analytics:

1. Add to `index.html` before `</head>`:
```html
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

2. Replace `G-XXXXXXXXXX` with your Analytics ID

---

## 🔒 Security Considerations

### What's Safe:
- ✅ All code runs in browser sandbox
- ✅ No server-side execution
- ✅ No database (uses localStorage)
- ✅ HTTPS by default (GitHub Pages)

### Important Notes:
- Don't commit API keys or secrets
- Use environment variables for sensitive data
- localStorage is per-domain (data stays local)

---

## 🎯 Success Checklist

After deployment, verify:

- [ ] Site loads at `https://YOUR-USERNAME.github.io/AI-IDE/`
- [ ] No console errors in browser DevTools
- [ ] Python code executes successfully
- [ ] Code editor (Monaco) works
- [ ] All panels open/close correctly
- [ ] Mobile version works
- [ ] GitHub Actions workflow is green ✅

---

## 📞 Need Help?

### Common Issues:
1. **Blank page** → Check vite.config.ts base path
2. **404 errors** → Verify GitHub Pages is enabled
3. **Python not working** → Check console for Pyodide errors
4. **Build fails** → Check Actions tab for error logs

### Resources:
- [Vite Deployment Guide](https://vitejs.dev/guide/static-deploy.html)
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Pyodide Documentation](https://pyodide.org/)

---

## 🎉 You're Done!

Your AI-IDE is now live on GitHub Pages! 🚀

**Next steps:**
1. Share your link with others
2. Add custom features
3. Star the repository
4. Contribute improvements

**Your live site:**
```
https://YOUR-USERNAME.github.io/AI-IDE/
```

---

*Happy coding!* 💻✨
