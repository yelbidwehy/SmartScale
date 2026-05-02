$spawnRate = 5
$hostUrl = "http://frontend.default.svc.cluster.local:80"

$testPlan = @(
    @{ Users = 100; Duration = 5  },
    @{ Users = 200; Duration = 7  },
    @{ Users = 300; Duration = 10 },
    @{ Users = 400; Duration = 12 },
    @{ Users = 500; Duration = 15 }
)

New-Item -ItemType Directory -Force -Path ".\datasets\raw" | Out-Null

foreach ($test in $testPlan) {

    $users = $test.Users
    $durationMinutes = $test.Duration

    Write-Host "======================================"
    Write-Host "Starting Locust run with $users users for $durationMinutes minutes"
    Write-Host "======================================"

    $startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Invoke-WebRequest -UseBasicParsing `
      -Uri "http://localhost:8089/swarm" `
      -Method POST `
      -Body @{
        user_count = $users
        spawn_rate = $spawnRate
        host = $hostUrl
      }

    Start-Sleep -Seconds ($durationMinutes * 60)

    $endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8089/stop" -Method GET


    python .\scripts\data_collection\extract_prometheus_logs.py `
        --start "$startTime" `
        --end "$endTime" `
        --run-name "run_${users}_users"

    Write-Host "Finished run with $users users"
    Write-Host ""

    Start-Sleep -Seconds 120
}

Write-Host "======================================"
Write-Host "Merging all runs into one dataset..."
Write-Host "======================================"

python .\scripts\data_preparation\merge_runs.py

Write-Host "Dataset ready: data\processed\smartscale_training_dataset.csv"