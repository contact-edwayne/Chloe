' start_ollama_hidden.vbs - launch start_ollama.bat with NO visible window.
'
' This is the target of the "chloe_ollama_serve" shortcut in the Windows
' Startup folder (run setup_ollama_startup.bat once to repoint it here).
' Replaces the old shortcut that ran start_ollama.bat directly and caused
' a console window to flash at every login.
'
' start_ollama.bat is itself idempotent, so running it this way is safe.

Dim fso, here
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
CreateObject("WScript.Shell").Run "cmd /c """ & here & "\start_ollama.bat""", 0, False
