@echo off
REM AI-IDE GitHub Pages Deployment Script (Windows)
REM Automatically builds and prepares for GitHub Pages deployment

echo.
echo ========================================
echo   AI-IDE GitHub Pages Deployment
echo ========================================
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo [Installing dependencies...]
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        exit /b 1
    )
    echo [OK] Dependencies installed
    echo.
)

REM Clean previous build
if exist "dist" (
    echo [Cleaning previous build...]
    rmdir /s /q dist
    echo [OK] Cleaned
    echo.
)

REM Build the project
echo [Building project...]
call npm run build

if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo [OK] Build successful!
echo.

REM Verify build
if exist "dist\index.html" (
    echo [OK] Build verified - index.html exists
) else (
    echo [ERROR] Build verification failed - index.html not found
    exit /b 1
)

echo.
echo ========================================
echo   Build complete and ready!
echo ========================================
echo.
echo Next steps:
echo   1. Commit and push to GitHub:
echo      git add .
echo      git commit -m "Deploy to GitHub Pages"
echo      git push origin main
echo.
echo   2. Wait 2-3 minutes for GitHub Actions
echo.
echo   3. Visit your site at:
echo      https://YOUR-USERNAME.github.io/AI-IDE/
echo.
echo Happy deploying!
echo.

pause
