import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // GitHub Pages configuration
  // Use './' for relative paths (works for both root and subdirectory deployments)
  base: './',
  
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    
    // Optimize chunk size
    rollupOptions: {
      output: {
        manualChunks: {
          'monaco': ['monaco-editor', '@monaco-editor/react'],
          'vendor': ['react', 'react-dom']
        }
      }
    }
  },
  
  optimizeDeps: {
    include: ['react', 'react-dom', 'monaco-editor']
  },
  
  // Handle SPA routing for GitHub Pages
  server: {
    port: 5173,
    open: true
  }
})
