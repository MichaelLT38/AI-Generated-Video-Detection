@echo off
REM Drag-and-drop an MP4 file onto this .bat to generate a spectrum MP4.
if "%~1"=="" (
    echo Drag an MP4 file onto this batch file.
    pause
    exit /b 1
)
python "%~dp0video_spectra.py" "%~1"
pause
