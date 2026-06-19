# Multi-run staged load experiment runner
# Reproduces fixed run names used by current dataset: train_01, train_02, train_03, test_01
# Uses staged Locust profiles and no keep-awake mouse movement logic.

$ErrorActionPreference = "Stop"

$hostUrl = "http://istio-gateway-istio.default.svc.cluster.local:80"
$locustBaseUrl = "http://localhost:8089"

# Base staged profile (same as run_load_experiments.ps1)
$baseStages = @(
    @{ Users = 30; Duration = 3 },
    @{ Users = 90; Duration = 4 },
    @{ Users = 130; Duration = 5 },
    @{ Users = 180; Duration = 8 },
    @{ Users = 130; Duration = 5 },
    @{ Users = 70; Duration = 4 },
    @{ Users = 30; Duration = 3 },
    @{ Users = 0; Duration = 1 }
)

# Run plans selected to match current exported runs in data/raw/prometheus_export
$runPlans = @(
    @{
        RunName   = "train_001"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 30; Duration = 3 },
            @{ Users = 90; Duration = 4 },
            @{ Users = 130; Duration = 5 },
            @{ Users = 180; Duration = 4 },
            @{ Users = 90; Duration = 4 },
            @{ Users = 30; Duration = 3 },
            @{ Users = 0; Duration = 1 }
        )
    },
    @{
        RunName   = "train_002"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 30; Duration = 3 },
            @{ Users = 90; Duration = 4 },
            @{ Users = 130; Duration = 5 },
            @{ Users = 180; Duration = 8 },
            @{ Users = 130; Duration = 5 },
            @{ Users = 70; Duration = 2 },
            @{ Users = 0; Duration = 1 }
        )
    },
    @{
        RunName   = "train_003"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 30; Duration = 3 },
            @{ Users = 90; Duration = 4 },
            @{ Users = 130; Duration = 5 },
            @{ Users = 180; Duration = 8 },
            @{ Users = 130; Duration = 3 },
            @{ Users = 90; Duration = 4 },
            @{ Users = 30; Duration = 3 },
            @{ Users = 0; Duration = 1 }
        )
    },
    @{
        RunName   = "train_004"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 30; Duration = 3 },
            @{ Users = 100; Duration = 4 },
            @{ Users = 200; Duration = 5 },
            @{ Users = 300; Duration = 8 },
            @{ Users = 200; Duration = 3 },
            @{ Users = 100; Duration = 4 },
            @{ Users = 50; Duration = 3 },
            @{ Users = 0; Duration = 1 }
        )
    },   
    @{
        RunName   = "train_005"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 30; Duration = 3 },
            @{ Users = 100; Duration = 4 },
            @{ Users = 200; Duration = 5 },
            @{ Users = 400; Duration = 8 },
            @{ Users = 500; Duration = 3 },
            @{ Users = 300; Duration = 4 },
            @{ Users = 100; Duration = 3 },
            @{ Users = 0; Duration = 1 }
        )
    },
    @{
        RunName   = "test_001"
        SpawnRate = 5
        Stages    = @(
            @{ Users = 30; Duration = 3 },
            @{ Users = 90; Duration = 4 },
            @{ Users = 130; Duration = 5 },
            @{ Users = 200; Duration = 8 },
            @{ Users = 300; Duration = 8 },
            @{ Users = 130; Duration = 5 },
            @{ Users = 70; Duration = 4 },
            @{ Users = 30; Duration = 3 },
            @{ Users = 0; Duration = 1 }
        )
    }
)

$allRunNames = @()
$successCount = 0
$failureCount = 0

Write-Host "Starting fixed staged collection for train/test runs"
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
        python .\scripts\data_collection\extract_prometheus_logs.py `
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

$runListFile = "data/raw/prometheus_export/_run_manifest.txt"
$allRunNames | Out-File -FilePath $runListFile -Force
Write-Host ""
Write-Host "Run manifest saved to: $runListFile"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Run: python .\scripts\data_collection\merge_runs.py"
Write-Host "2. Output will be in: data/processed/"
