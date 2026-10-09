@echo off
REM wiki_maintain.bat -- nightly via Task Scheduler. Collects finished Claude
REM batches into brain\proposals\ and submits new duplicate clusters.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python wiki_maintain.py run --limit 25 >> wiki_maintain.log 2>&1
