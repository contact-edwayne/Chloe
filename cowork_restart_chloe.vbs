' cowork_restart_chloe.vbs - stop + restart Chloe's background services with
' no prompts and no visible windows. Written by Cowork/Claude so a restart
' can be triggered with a single double-click (no terminal typing needed).
Dim fso, here
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
CreateObject("WScript.Shell").Run "cmd /c taskkill /im python.exe /f & timeout /t 2 /nobreak & """ & here & "\start_chloe_full.bat""", 0, False
