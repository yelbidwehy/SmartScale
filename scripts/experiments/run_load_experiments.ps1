$hostUrl = "http://istio-gateway-istio.default.svc.cluster.local:80"
$spawnRate = 10
$runName = "run_ramp_down_300_200_100_0"

$stages = @(
    @{ Users = 200; Duration = 5 },
    @{ Users = 100; Duration = 5 },
    @{ Users = 0;   Duration = 5 }
)

$startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

foreach ($stage in $stages) {
    $users = $stage.Users
    $durationMinutes = $stage.Duration

    Write-Host "Setting Locust users to $users for $durationMinutes minutes"

    if ($users -eq 0) {
        Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8089/stop" -Method GET
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

if ([string]::IsNullOrWhiteSpace($runName)) {
    $stageNames = ($stages | ForEach-Object { $_.Users }) -join "_"
    $runName = "run_${stageNames}"
}

python .\scripts\data_collection\extract_prometheus_logs.py `
    --start "$startTime" `
    --end "$endTime" `
    --run-name $runName