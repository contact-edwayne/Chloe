@echo off
REM Run all 13 chloe_jobs in sequence. Captures pass/fail per job + total time.
REM Use once after register_chloe_jobs.ps1 to verify the migration end-to-end.
REM Output to stdout AND to logs\verify_<timestamp>.log.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set STAMP=%DATE:/=-%_%TIME::=-%
set STAMP=%STAMP: =0%
set STAMP=%STAMP:.=-%
set LOG=logs\verify_%STAMP%.log
if not exist logs mkdir logs

echo === verify chloe_jobs ===                              > "%LOG%" 2>&1
echo Started: %DATE% %TIME%                                 >> "%LOG%" 2>&1
echo. >> "%LOG%" 2>&1
echo === verify chloe_jobs ===
echo Log: %LOG%
echo.

REM Ordered cheap -^> expensive so failures surface fast.
REM Quote-wrapped to handle spaces if anyone moves the folder.
set "JOBS=weekly-autonomous-audit daily-journal-stub daily-cowork-fact-extract daily-voice-persona-mining weekly-persona-drift weekly-persona-evolution daily-topic-rotation daily-critical-thinking-exercise daily-finance-ingest daily-morning-brief weekly-cross-domain-synthesis friday-meta-review weekly-backup"

set /a TOTAL=0
set /a OK_COUNT=0
set /a FAIL_COUNT=0
set FAIL_LIST=

for %%j in (%JOBS%) do (
    set /a TOTAL+=1
    echo --- !TOTAL!. %%j ---
    echo --- !TOTAL!. %%j --- >> "%LOG%" 2>&1
    call chloe_jobs.bat %%j >> "%LOG%" 2>&1
    if !errorlevel! equ 0 (
        set /a OK_COUNT+=1
        echo   OK
    ) else (
        set /a FAIL_COUNT+=1
        set "FAIL_LIST=!FAIL_LIST! %%j"
        echo   FAIL ^(exit code !errorlevel!^)
    )
)

echo.
echo === SUMMARY ===
echo. >> "%LOG%" 2>&1
echo === SUMMARY === >> "%LOG%" 2>&1
echo total : %TOTAL%
echo total : %TOTAL% >> "%LOG%" 2>&1
echo ok    : %OK_COUNT%
echo ok    : %OK_COUNT% >> "%LOG%" 2>&1
echo fail  : %FAIL_COUNT%
echo fail  : %FAIL_COUNT% >> "%LOG%" 2>&1
if not "%FAIL_LIST%"=="" (
    echo failed:%FAIL_LIST%
    echo failed:%FAIL_LIST% >> "%LOG%" 2>&1
)
echo Finished: %DATE% %TIME%
echo Finished: %DATE% %TIME% >> "%LOG%" 2>&1
echo.
echo Full log: %LOG%
echo Paste the SUMMARY block + any FAIL lines back to Claude for diagnosis.
endlocal
