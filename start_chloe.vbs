' start_chloe.vbs - DOUBLE-CLICK THIS to launch Chloe with no console windows.
'
' This is the new one-click launcher. It runs start_chloe_full.bat hidden;
' that script does the Tailscale/Ollama sanity checks and then launches the
' backend, static file server and wiki watcher as hidden background
' processes (see svc.bat + run_hidden.vbs).
'
' Result: nothing visible except the Chloe desktop HUD a few seconds later.
'
' To see service logs:  run show_chloe_logs.bat
' To stop Chloe:        run stop_chloe.bat

Dim fso, here
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
CreateObject("WScript.Shell").Run "cmd /c """ & here & "\start_chloe_full.bat""", 0, False
