$hostUrl = "http://istio-gateway-istio.default.svc.cluster.local:80"
$spawnRate = 5
$runName = "test_01"

$stages = @(
    @{ Users = 30;  Duration = 3 },
    @{ Users = 90;  Duration = 4 },
    @{ Users = 130; Duration = 5 },
    @{ Users = 180; Duration = 8 },
    @{ Users = 130; Duration = 5 },
    @{ Users = 70;  Duration = 4 },
    @{ Users = 30;  Duration = 3 },
    @{ Users = 0;   Duration = 1 }
)
$startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

foreach ($stage in $stages) {
    $users = $stage.Users
    $durationMinutes = $stage.Duration

    Write-Host "========================================"
    Write-Host "Setting Locust users to $users for $durationMinutes minutes"
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "========================================"

    if ($users -eq 0) {
        Invoke-WebRequest -UseBasicParsing `
            -Uri "http://localhost:8089/stop" `
            -Method GET
    }
    else {
        Invoke-WebRequest -UseBasicParsing `
            -Uri "http://localhost:8089/swarm" `
            -Method POST `
            -Body @{
                user_count = $users
                spawn_rate = $spawnRate
                host = $hostUrl
            }
    }

    Start-Sleep -Seconds ($durationMinutes * 60)
}

$endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

python .\scripts\data_collection\extract_prometheus_logs.py `
    --start "$startTime" `
    --end "$endTime" `
    --run-name $runName `
    --step 5s