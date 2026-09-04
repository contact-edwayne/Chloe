' run_hidden.vbs - run a command line with NO visible window.
'
' Usage:  wscript run_hidden.vbs "<command line>"
'
' This is the generic "make it invisible" helper for Chloe's launchers.
' WScript.Shell.Run with window style 0 starts the process fully detached
' (it is NOT tied to the caller's console), so the spawned process keeps
' running after this script and its caller exit.
'
' Used by start_chloe_full.bat to launch the backend, static server and
' wiki watcher without console windows.

If WScript.Arguments.Count > 0 Then
    CreateObject("WScript.Shell").Run WScript.Arguments(0), 0, False
End If
