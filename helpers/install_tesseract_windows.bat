@echo off
echo ========================================
echo    Tesseract OCR Installation Script
echo ========================================
echo.

echo Checking if Chocolatey is installed...
choco --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Chocolatey is not installed. Installing Chocolatey first...
    echo.
    echo Please run this script as Administrator (Right-click -> Run as Administrator)
    echo.
    pause
    exit /b 1
)

echo Chocolatey is installed. Installing Tesseract...
echo.

choco install tesseract -y

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo    Tesseract installed successfully!
    echo ========================================
    echo.
    echo Please restart your terminal/command prompt for PATH changes to take effect.
    echo.
    echo To verify installation, run: tesseract --version
    echo.
) else (
    echo.
    echo ========================================
    echo    Installation failed!
    echo ========================================
    echo.
    echo Please try installing manually:
    echo 1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo 2. Run the installer as Administrator
    echo 3. Make sure to add Tesseract to PATH during installation
    echo.
)

pause
