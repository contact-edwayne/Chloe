# register_chloe_jobs.ps1
# Creates Windows Task Scheduler entries for every chloe_jobs.py job.
# Runs as the current user (non-elevated). Mirrors the Cowork schedules.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File register_chloe_jobs.ps1
#   powershell -ExecutionPolicy Bypass -File register_chloe_jobs.ps1 -Remove
#
# After registration, run any task manually once to verify:
#   schtasks /run /tn "Chloe\daily-journal-stub"
# Then disable the corresponding Cowork task to avoid duplicate output.

param(
  [switch]$Remove
)

$ErrorActionPreference = "Stop"

$here   = Split-Path -Parent $MyInvocation.MyCommand.Path
$batch  = Join-Path $here "chloe_jobs.bat"
$logDir = Join-Path $here "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# (TaskName, JobName, Trigger description, Time string, Days-of-week or "Daily" or "Weekly:<dow>")
# Triggers are in the user's local time, matching the Cowork crons.
$jobs = @(
  @{ Name="daily-journal-stub";              Job="daily-journal-stub";              Schedule="Daily";   Time="23:00" },
  @{ Name="daily-cowork-fact-extract";       Job="daily-cowork-fact-extract";       Schedule="Daily";   Time="22:30" },
  @{ Name="daily-voice-persona-mining";      Job="daily-voice-persona-mining";      Schedule="Daily";   Time="22:00" },
  @{ Name="daily-arcade-distill";            Job="daily-arcade-distill";            Schedule="Daily";   Time="22:15" },
  @{ Name="daily-topic-rotation";            Job="daily-topic-rotation";            Schedule="Weekly";  Time="06:00"; Days="MON,TUE,WED,THU,FRI,SAT" },
  @{ Name="daily-finance-ingest";            Job="daily-finance-ingest";            Schedule="Weekly";  Time="07:30"; Days="MON,TUE,WED,THU,FRI" },
  @{ Name="daily-morning-brief";             Job="daily-morning-brief";             Schedule="Daily";   Time="07:00" },
  @{ Name="daily-critical-thinking-exercise"; Job="daily-critical-thinking-exercise"; Schedule="Weekly";  Time="13:00"; Days="MON,TUE,WED,THU,FRI" },
  @{ Name="weekly-backup";                   Job="weekly-backup";                   Schedule="Weekly";  Time="03:00"; Days="SUN" },
  @{ Name="weekly-autonomous-audit";         Job="weekly-autonomous-audit";         Schedule="Weekly";  Time="04:00"; Days="SUN" },
  @{ Name="weekly-persona-drift";            Job="weekly-persona-drift";            Schedule="Weekly";  Time="05:00"; Days="SUN" },
  @{ Name="weekly-persona-evolution";        Job="weekly-persona-evolution";        Schedule="Weekly";  Time="06:00"; Days="SUN" },
  @{ Name="weekly-cross-domain-synthesis";   Job="weekly-cross-domain-synthesis";   Schedule="Weekly";  Time="09:00"; Days="SUN" },
  @{ Name="friday-meta-review";              Job="friday-meta-review";              Schedule="Weekly";  Time="08:00"; Days="FRI" },
  # Stage-4 autonomous code-fix proposer. DRAFT-ONLY while the autonomous enable
  # flag is OFF (the job scans recurring tracebacks + writes proposals but applies
  # nothing unless enabled). Added 2026-05-25 after the formal watchdog_watch=
  # healthy capture; leave enable OFF to soak the draft path on real errors before
  # trusting unattended apply. To go apply-capable: mcp__chloe__autonomous_set_enabled.
  @{ Name="autonomous-fix-recurring-errors"; Job="autonomous-fix-recurring-errors"; Schedule="Daily";   Time="04:00" }
)

if ($Remove) {
  Write-Host "Removing Chloe\* scheduled tasks..." -ForegroundColor Yellow
  foreach ($j in $jobs) {
    $tn = "Chloe\$($j.Name)"
    try {
      schtasks /delete /tn $tn /f | Out-Null
      Write-Host "  removed: $tn" -ForegroundColor DarkGray
    } catch {
      Write-Host "  not present: $tn" -ForegroundColor DarkGray
    }
  }
  exit 0
}

if (-not (Test-Path $batch)) {
  Write-Host "ERROR: $batch not found. Aborting." -ForegroundColor Red
  exit 2
}

Write-Host "Registering Chloe scheduled tasks under \Chloe\..." -ForegroundColor Green
Write-Host "Batch: $batch"
Write-Host ""

foreach ($j in $jobs) {
  $tn = "Chloe\$($j.Name)"
  $tr = "`"$batch`" $($j.Job)"

  # Build the schtasks command.
  $args = @(
    "/create", "/f",
    "/tn", $tn,
    "/tr", $tr,
    "/sc", $j.Schedule,
    "/st", $j.Time,
    "/it"  # interactive — runs when user is logged on, matches Cowork's "while app is open"
  )
  if ($j.ContainsKey("Days") -and $j.Days) {
    $args += @("/d", $j.Days)
  }

  $daysStr = if ($j.ContainsKey("Days") -and $j.Days) { $j.Days } else { "" }
  Write-Host ("  {0,-40} {1,-8} {2} {3}" -f $j.Name, $j.Schedule, $j.Time, $daysStr) -NoNewline
  try {
    & schtasks @args 2>&1 | Out-Null
    Write-Host "  OK" -ForegroundColor Green
  } catch {
    Write-Host "  FAILED: $_" -ForegroundColor Red
  }
}

Write-Host ""
Write-Host "Done. To list:    schtasks /query /tn `"Chloe\*`"" -ForegroundColor Cyan
Write-Host "To run one now:   schtasks /run /tn `"Chloe\daily-journal-stub`"" -ForegroundColor Cyan
Write-Host "To remove all:    powershell -File register_chloe_jobs.ps1 -Remove" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Run each task manually first to verify output:" -ForegroundColor Yellow
Write-Host "       chloe_jobs.bat daily-journal-stub" -ForegroundColor DarkYellow
Write-Host "  2. Inspect logs\chloe_jobs.log for errors." -ForegroundColor Yellow
Write-Host "  3. Once a task's local version is verified, DISABLE the matching" -ForegroundColor Yellow
Write-Host "     Cowork task to avoid duplicate output (e.g. via Cowork's UI" -ForegroundColor Yellow
Write-Host "     or mcp__scheduled-tasks__update_scheduled_task with enabled=false)." -ForegroundColor Yellow
