@echo off
cd /d "%~dp0"
rem Compatibility alias: main's regular configuration now selects CUDA.
rem Do not load old trial profiles containing interpreter settings from another branch.
call "%~dp0start-panel.cmd" %*
