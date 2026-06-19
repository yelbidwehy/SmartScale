# Multi-run staged load experiment runner — low volatility batch
# Uses 8 train + 2 test runs designed to reduce step-to-step RPS volatility
# compared to the original train_001-004/test_001 batch, while keeping
# test run peaks and minimums nested inside the train runs' range
# (verified: max train peak 260 >= max test peak 195; min train value
# 15 <= min test value 45 — no scaler extrapolation risk).

$ErrorActionPreference = "Stop"

# ── Resolve project root explicitly ───────────────────────────────────────
# Fixes the path bug from the original script: $runListFile and the python
# call below used relative paths, which resolved relative to wherever this
# script was invoked from (scripts\experiments\) instead of the project
# root, causing "scripts\experiments\scripts\data_collection\..." errors
# and a missing-directory failure when writing the manifest file.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

$PythonExtractScript = Join-Path $ProjectRoot "scripts\data_collection\extract_prometheus_logs.py"
$RunManifestDir = Join-Path $ProjectRoot "data\raw\prometheus_export"
$RunListFile = Join-Path $RunManifestDir "_run_manifest_lowvol.txt"

if (-not (Test-Path $RunManifestDir)) {
    New-Item -ItemType Directory -Path $RunManifestDir -Force | Out-Null
}

$hostUrl = "http://istio-gateway-istio.default.svc.cluster.local:80"
$locustBaseUrl = "http://localhost:8089"

# ── Run plans: low-volatility batch, verified ranges ──────────────────────
# Max single-stage jump across all runs is 50 users (vs up to 100 in the
# original batch). Stage durations stretched slightly (4-8 min) to let RPS
# settle before each transition.

$runPlans = @(
    @{
        RunName   = "train_lv_001"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 20;  Duration = 4 },
            @{ Users = 60;  Duration = 5 },
            @{ Users = 100; Duration = 5 },
            @{ Users = 100; Duration = 5 },
            @{ Users = 60;  Duration = 4 },
            @{ Users = 20;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "train_lv_002"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 30;  Duration = 4 },
            @{ Users = 75;  Duration = 5 },
            @{ Users = 120; Duration = 6 },
            @{ Users = 120; Duration = 5 },
            @{ Users = 75;  Duration = 4 },
            @{ Users = 30;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "train_lv_003"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 40;  Duration = 4 },
            @{ Users = 85;  Duration = 5 },
            @{ Users = 130; Duration = 6 },
            @{ Users = 170; Duration = 6 },
            @{ Users = 130; Duration = 5 },
            @{ Users = 85;  Duration = 4 },
            @{ Users = 40;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "train_lv_004"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 25;  Duration = 4 },
            @{ Users = 65;  Duration = 5 },
            @{ Users = 110; Duration = 5 },
            @{ Users = 110; Duration = 5 },
            @{ Users = 150; Duration = 5 },
            @{ Users = 110; Duration = 4 },
            @{ Users = 65;  Duration = 4 },
            @{ Users = 25;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "train_lv_005"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 50;  Duration = 4 },
            @{ Users = 95;  Duration = 5 },
            @{ Users = 140; Duration = 6 },
            @{ Users = 185; Duration = 6 },
            @{ Users = 220; Duration = 6 },
            @{ Users = 185; Duration = 5 },
            @{ Users = 140; Duration = 4 },
            @{ Users = 95;  Duration = 4 },
            @{ Users = 50;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "train_lv_006"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 15;  Duration = 5 },
            @{ Users = 45;  Duration = 5 },
            @{ Users = 80;  Duration = 5 },
            @{ Users = 80;  Duration = 6 },
            @{ Users = 45;  Duration = 4 },
            @{ Users = 15;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "train_lv_007"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 60;  Duration = 4 },
            @{ Users = 110; Duration = 5 },
            @{ Users = 160; Duration = 6 },
            @{ Users = 210; Duration = 6 },
            @{ Users = 260; Duration = 6 },
            @{ Users = 210; Duration = 5 },
            @{ Users = 160; Duration = 5 },
            @{ Users = 110; Duration = 4 },
            @{ Users = 60;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "train_lv_008"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 35;  Duration = 4 },
            @{ Users = 70;  Duration = 5 },
            @{ Users = 105; Duration = 5 },
            @{ Users = 140; Duration = 5 },
            @{ Users = 105; Duration = 5 },
            @{ Users = 70;  Duration = 4 },
            @{ Users = 35;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "test_lv_001"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 45;  Duration = 4 },
            @{ Users = 90;  Duration = 5 },
            @{ Users = 135; Duration = 6 },
            @{ Users = 135; Duration = 6 },
            @{ Users = 90;  Duration = 4 },
            @{ Users = 45;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    },
    @{
        RunName   = "test_lv_002"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 55;  Duration = 4 },
            @{ Users = 100; Duration = 5 },
            @{ Users = 150; Duration = 6 },
            @{ Users = 195; Duration = 6 },
            @{ Users = 150; Duration = 5 },
            @{ Users = 100; Duration = 4 },
            @{ Users = 55;  Duration = 3 },
            @{ Users = 0;   Duration = 1 }
        )
    }
)

$allRunNames = @()
$successCount = 0
$failureCount = 0

Write-Host "Starting low-volatility staged collection for train/test runs"
Write-Host "Runs to execute: $($runPlans.Count)"
Write-Host ""

function Invoke-LocustApi {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Method,
        [hashtable]$Body
    )

    $params = @{
        UseBasicParsing = $true
        Uri             = $Uri
        Method          = $Method
    }

    if ($Body) {
        $params.Body = $Body
    }

    $response = Invoke-WebRequest @params
    if (-not ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)) {
        throw "Locust API call failed: $Method $Uri returned status $($response.StatusCode)"
    }

    return $response
}

foreach ($plan in $runPlans) {
    $runName = $plan.RunName
    $spawnRate = $plan.SpawnRate
    $stages = $plan.Stages

    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "Run: $runName"
    Write-Host "Spawn Rate: $spawnRate/s"
    Write-Host "Stages: $($stages.Count)"
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    try {
        $startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        foreach ($stage in $stages) {
            $users = $stage.Users
            $durationMinutes = $stage.Duration

            Write-Host "Setting users to $users for $durationMinutes minutes..."

            if ($users -eq 0) {
                Invoke-LocustApi -Uri "$locustBaseUrl/stop" -Method GET
            }
            else {
                Invoke-LocustApi `
                    -Uri "$locustBaseUrl/swarm" `
                    -Method POST `
                    -Body @{
                    user_count = $users
                    spawn_rate = $spawnRate
                    host       = $hostUrl
                }
            }

            Start-Sleep -Seconds ($durationMinutes * 60)
        }

        # Ensure load is stopped even when final stage does not include users=0
        Invoke-LocustApi -Uri "$locustBaseUrl/stop" -Method GET

        $endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        Write-Host "Extracting Prometheus metrics for $runName..."
        python $PythonExtractScript `
            --start "$startTime" `
            --end "$endTime" `
            --run-name $runName `
            --step 5s

        Write-Host "[OK] Complete: $runName" -ForegroundColor Green
        $allRunNames += $runName
        $successCount++
    }
    catch {
        Write-Host "[FAIL] Failed: $runName" -ForegroundColor Red
        Write-Host "Error: $_"

        try {
            Invoke-LocustApi -Uri "$locustBaseUrl/stop" -Method GET | Out-Null
        }
        catch {
            Write-Host "Warning: failed to stop Locust after error." -ForegroundColor Yellow
        }

        $failureCount++
    }

    Write-Host "Cooling down for 30 seconds before next run..."
    Start-Sleep -Seconds 30
}

Write-Host ""
Write-Host "====== Data Collection Complete ======" -ForegroundColor Cyan
Write-Host "Total successful: $successCount / $($runPlans.Count)"
Write-Host "Total failed: $failureCount / $($runPlans.Count)"
Write-Host ""
Write-Host "Collected runs:"
foreach ($run in $allRunNames) {
    Write-Host "  - $run"
}

$allRunNames | Out-File -FilePath $RunListFile -Force
Write-Host ""
Write-Host "Run manifest saved to: $RunListFile"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Run: python .\scripts\data_collection\merge_runs.py"
Write-Host "2. Output will be in: data/processed/"