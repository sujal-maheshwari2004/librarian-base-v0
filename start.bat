@echo off
title Kill Non-Essential Processes
color 0C

echo ==========================================
echo   Freeing RAM for ML / SLM Training
echo ==========================================
echo.
echo WARNING: This will close running apps.
echo Save your work first.
echo.
pause

echo.
echo --- Killing browsers ---
taskkill /F /IM chrome.exe >nul 2>&1
taskkill /F /IM msedge.exe >nul 2>&1
taskkill /F /IM firefox.exe >nul 2>&1
taskkill /F /IM brave.exe >nul 2>&1
taskkill /F /IM opera.exe >nul 2>&1
taskkill /F /IM comet.exe >nul 2>&1

echo --- Killing chat / meeting apps ---
taskkill /F /IM discord.exe >nul 2>&1
taskkill /F /IM slack.exe >nul 2>&1
taskkill /F /IM teams.exe >nul 2>&1
taskkill /F /IM zoom.exe >nul 2>&1
taskkill /F /IM skype.exe >nul 2>&1
taskkill /F /IM whatsapp.exe >nul 2>&1

echo --- Killing game launchers ---
taskkill /F /IM steam.exe >nul 2>&1
taskkill /F /IM epicgameslauncher.exe >nul 2>&1
taskkill /F /IM battle.net.exe >nul 2>&1
taskkill /F /IM riotclientservices.exe >nul 2>&1
taskkill /F /IM ubisoftconnect.exe >nul 2>&1

echo --- Killing Adobe background services ---
taskkill /F /IM creativecloud.exe >nul 2>&1
taskkill /F /IM adobeipcbroker.exe >nul 2>&1
taskkill /F /IM ccxprocess.exe >nul 2>&1
taskkill /F /IM cclibrary.exe >nul 2>&1
taskkill /F /IM coreSync.exe >nul 2>&1

echo --- Killing cloud sync ---
taskkill /F /IM onedrive.exe >nul 2>&1
taskkill /F /IM dropbox.exe >nul 2>&1
taskkill /F /IM googledrivesync.exe >nul 2>&1

echo --- Killing misc Windows junk ---
taskkill /F /IM yourphone.exe >nul 2>&1
taskkill /F /IM widgetservice.exe >nul 2>&1
taskkill /F /IM searchapp.exe >nul 2>&1
taskkill /F /IM cortana.exe >nul 2>&1
taskkill /F /IM xboxapp.exe >nul 2>&1
taskkill /F /IM xboxgamingoverlay.exe >nul 2>&1

echo --- Killing updater services ---
taskkill /F /IM googleupdate.exe >nul 2>&1
taskkill /F /IM microsoftedgeupdate.exe >nul 2>&1

echo.
echo ==========================================
echo   Done.
echo   Windows core services untouched.
echo   Start your training loop now.
echo ==========================================
echo.
pause

echo Starting training with HIGH priority...
start "" /HIGH python train.py
