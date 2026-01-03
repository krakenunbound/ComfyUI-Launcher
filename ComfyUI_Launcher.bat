@echo off
REM ComfyUI Graphical Launcher
REM Opens a web-based GUI in your browser to control ComfyUI
REM No command prompt window for ComfyUI - everything in the browser!

start "" /B "%~dp0python_embeded\pythonw.exe" "%~dp0ComfyUI_Launcher.pyw"
