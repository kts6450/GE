@echo off
REM Raw "n8n start" uses 5678 -> TCube conflict on this PC. This runs n8n\start-n8n.ps1 (auto port).
cd /d "%~dp0"
call "%~dp0n8n\start-n8n.cmd"
