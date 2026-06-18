# Multi-scenario load experiment runner
# Collects data from multiple user load scenarios for LSTM training
# Target: 2500+ records total (2000 train, 500 test)
# Each 10-minute run produces ~121 samples at 5s intervals

$ErrorActionPreference = "Stop"

$hostUrl = "http://istio-gateway-istio.default.svc.cluster.local:80"
$locustBaseUrl = "http://localhost:8089"
$rateWindow = "1m"
$baseTag = "train"

# Define load scenarios with different user counts and ramp rates
# Roughly 22 runs × 121 samples = 2662 total records for TRAINING
$trainingScenarios = @(
    @{ Users = 50;   SpawnRate = 5;   Name = "low_load_5_ramp" },
    @{ Users = 100;  SpawnRate = 10;  Name = "med_load_10_ramp" },
    @{ Users = 150;  SpawnRate = 15;  Name = "med_high_15_ramp" },
    @{ Users = 200;  SpawnRate = 20;  Name = "high_load_20_ramp" },
    @{ Users = 300;  SpawnRate = 10;  Name = "peak_10_ramp" },
    @{ Users = 300;  SpawnRate = 20;  Name = "peak_20_ramp" },
    @{ Users = 250;  SpawnRate = 15;  Name = "sustained_250" },
    @{ Users = 100;  SpawnRate = 5;   Name = "low_slow_ramp" },
    @{ Users = 200;  SpawnRate = 10;  Name = "mid_stable" },
    @{ Users = 50;   SpawnRate = 10;  Name = "light_load" },
    @{ Users = 150;  SpawnRate = 20;  Name = "medium_fast" },
    @{ Users = 300;  SpawnRate = 15;  Name = "peak_moderate" },
    @{ Users = 120;  SpawnRate = 8;   Name = "varied_120" },
    @{ Users = 180;  SpawnRate = 12;  Name = "varied_180" },
    @{ Users = 280;  SpawnRate = 25;  Name = "high_aggressive" },
    @{ Users = 80;   SpawnRate = 6;   Name = "low_conservative" },
    @{ Users = 220;  SpawnRate = 18;  Name = "mid_push" },
    @{ Users = 160;  SpawnRate = 11;  Name = "balanced_160" },
    @{ Users = 320;  SpawnRate = 16;  Name = "extreme_320" },
    @{ Users = 90;   SpawnRate = 7;   Name = "light_90" },
    @{ Users = 240;  SpawnRate = 22;  Name = "high_240" },
    @{ Users = 300;  SpawnRate = 30;  Name = "peak_fast_ramp" }
)

# Define testing scenarios (different patterns, ~5 runs × 121 = 605 records for TESTING)
# These are deliberately different from training to avoid leakage
$testingScenarios = @(
    @{ Users = 175;  SpawnRate = 14;  Name = "test_hybrid_175" },
    @{ Users = 260;  SpawnRate = 19;  Name = "test_varied_260" },
    @{ Users = 130;  SpawnRate = 9;   Name = "test_medium_130" },
    @{ Users = 290;  SpawnRate = 23;  Name = "test_peak_290" },
    @{ Users = 110;  SpawnRate = 8;   Name = "test_low_110" }
)

# Combine all scenarios
$allScenarios = @(
    @{ Type = "train"; Scenarios = $trainingScenarios },
    @{ Type = "test"; Scenarios = $testingScenarios }
)

$allRunNames = @()
$successCount = 0
$failureCount = 0

Write-Host "Starting combined training and testing load test collection"
Write-Host "Training scenarios: $($trainingScenarios.Count) (~2662 records)"
Write-Host "Testing scenarios: $($testingScenarios.Count) (~605 records)"
Write-Host "Total expected records: ~3267"
Write-Host ""

function Invoke-LocustApi {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Method,
        [hashtable]$Body
    )

    $params = @{
        UseBasicParsing = $true
        Uri = $Uri
        Method = $Method
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

function Keep-SystemAwake {
    param(
        [Parameter(Mandatory = $true)][int]$DurationSeconds,
        [int]$JiggleIntervalSeconds = 30
    )
    
    # Keep system awake by moving mouse periodically
    [int]$elapsed = 0
    
    while ($elapsed -lt $DurationSeconds) {
        [int]$remaining = $DurationSeconds - $elapsed
        [int]$sleepTime = if ($JiggleIntervalSeconds -lt $remaining) { $JiggleIntervalSeconds } else { $remaining }
        
        # Sleep for the interval
        Start-Sleep -Seconds $sleepTime
        
        # Move mouse slightly to prevent lock
        if ($remaining -gt 0) {
            Add-Type -AssemblyName System.Windows.Forms
            Add-Type -AssemblyName System.Drawing
            $pos = [System.Windows.Forms.Cursor]::Position
            [int]$curX = $pos.X
            [int]$curY = $pos.Y
            [int]$newX = $curX + 1
            [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point($newX, $curY)
            Start-Sleep -Milliseconds 50
            [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point($curX, $curY)
        }
        
        $elapsed = $elapsed + $sleepTime
    }
}

# Process training scenarios first
Write-Host "============================================" -ForegroundColor Green
Write-Host "PHASE 1: TRAINING DATA COLLECTION"
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

foreach ($scenario in $trainingScenarios) {
    $scenarioName = $scenario.Name
    $users = $scenario.Users
    $spawnRate = $scenario.SpawnRate
    $runName = "train_{0}_{1}" -f $scenarioName, (Get-Date -Format "yyyyMMdd_HHmmss")
    
    Write-Host "Scenario: $scenarioName" -ForegroundColor Cyan
    Write-Host "Users: $users | Spawn Rate: $spawnRate/s"
    Write-Host "Run: $runName"
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    try {
        $startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        Write-Host "Starting Locust swarm..."
        Invoke-LocustApi `
            -Uri "$locustBaseUrl/swarm" `
            -Method POST `
            -Body @{
                user_count = $users
                spawn_rate = $spawnRate
                host = $hostUrl
            }

        Write-Host "Running for 10 minutes (keeping system awake)..."
        Keep-SystemAwake -DurationSeconds 600 -JiggleIntervalSeconds 30

        Write-Host "Stopping Locust..."
        Invoke-LocustApi `
            -Uri "$locustBaseUrl/stop" `
            -Method GET

        $endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        Write-Host "Extracting Prometheus metrics..."
        python .\scripts\data_collection\extract_prometheus_logs.py `
            --start "$startTime" `
            --end "$endTime" `
            --run-name $runName `
            --step 5s `
            --rate-window 1m

        Write-Host "[OK] Complete: $runName" -ForegroundColor Green
        $allRunNames += $runName
        $successCount++

    }
    catch {
        Write-Host "[FAIL] Failed: $scenarioName" -ForegroundColor Red
        Write-Host "Error: $_"
        $failureCount++
    }

    Write-Host "Waiting 30s before next scenario..."
    Keep-SystemAwake -DurationSeconds 30 -JiggleIntervalSeconds 10
}

# Process testing scenarios
Write-Host ""
Write-Host "============================================" -ForegroundColor Yellow
Write-Host "PHASE 2: TESTING DATA COLLECTION"
Write-Host "============================================" -ForegroundColor Yellow
Write-Host ""

foreach ($scenario in $testingScenarios) {
    $scenarioName = $scenario.Name
    $users = $scenario.Users
    $spawnRate = $scenario.SpawnRate
    $runName = "test_{0}_{1}" -f $scenarioName, (Get-Date -Format "yyyyMMdd_HHmmss")
    
    Write-Host "Scenario: $scenarioName" -ForegroundColor Cyan
    Write-Host "Users: $users | Spawn Rate: $spawnRate/s"
    Write-Host "Run: $runName"
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    try {
        $startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        Write-Host "Starting Locust swarm..."
        Invoke-LocustApi `
            -Uri "$locustBaseUrl/swarm" `
            -Method POST `
            -Body @{
                user_count = $users
                spawn_rate = $spawnRate
                host = $hostUrl
            }

        Write-Host "Running for 10 minutes (keeping system awake)..."
        Keep-SystemAwake -DurationSeconds 600 -JiggleIntervalSeconds 30

        Write-Host "Stopping Locust..."
        Invoke-LocustApi `
            -Uri "$locustBaseUrl/stop" `
            -Method GET

        $endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        Write-Host "Extracting Prometheus metrics..."
        python .\scripts\data_collection\extract_prometheus_logs.py `
            --start "$startTime" `
            --end "$endTime" `
            --run-name $runName `
            --step 5s `
            --rate-window 1m

        Write-Host "[OK] Complete: $runName" -ForegroundColor Green
        $allRunNames += $runName
        $successCount++

    }
    catch {
        Write-Host "[FAIL] Failed: $scenarioName" -ForegroundColor Red
        Write-Host "Error: $_"
        $failureCount++
    }

    Write-Host "Waiting 30s before next scenario..."
    Keep-SystemAwake -DurationSeconds 30 -JiggleIntervalSeconds 10
}

Write-Host ""
Write-Host "====== Data Collection Complete ======" -ForegroundColor Cyan
Write-Host "Total successful: $successCount / $($trainingScenarios.Count + $testingScenarios.Count)"
Write-Host "Total failed: $failureCount / $($trainingScenarios.Count + $testingScenarios.Count)"
Write-Host ""
Write-Host "Collected runs:"
foreach ($run in $allRunNames) {
    Write-Host "  - $run"
}

# Save run list for downstream processing
$runListFile = "data/raw/prometheus_export/_run_manifest.txt"
$allRunNames | Out-File -FilePath $runListFile -Force
Write-Host ""
Write-Host "Run manifest saved to: $runListFile"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Run: python .\scripts\data_collection\merge_runs.py"
Write-Host "2. This will merge all train_* and test_* runs automatically"
Write-Host "3. Output will be in: data/processed/"
