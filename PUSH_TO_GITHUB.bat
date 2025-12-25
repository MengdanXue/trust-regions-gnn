@echo off
echo ===============================================
echo   Trust Regions GNN - Push to GitHub
echo ===============================================
echo.

REM Check if repository exists on GitHub first
echo Please make sure you have created the repository:
echo   https://github.com/MengdanXue/trust-regions-gnn
echo.
echo Press any key to continue after creating the repo...
pause > nul

REM Configure git
git config --global http.postBuffer 524288000

REM Add remote if not exists
git remote remove origin 2>nul
git remote add origin https://github.com/MengdanXue/trust-regions-gnn.git

REM Rename branch to main
git branch -M main

REM Push to GitHub
echo.
echo Pushing to GitHub...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ===============================================
    echo   SUCCESS! Code pushed to GitHub
    echo   https://github.com/MengdanXue/trust-regions-gnn
    echo ===============================================
) else (
    echo ===============================================
    echo   Push failed. Please check:
    echo   1. Repository exists on GitHub
    echo   2. You have write access
    echo   3. Network connection is stable
    echo   4. Try using VPN if in China
    echo ===============================================
)

pause
