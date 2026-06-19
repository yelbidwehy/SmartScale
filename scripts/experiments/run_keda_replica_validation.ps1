# Real KEDA validation run
# Generates staged load and records deployment/HPA replica behavior over time.

param(
    [string]$Namespace = "default",
    [string]$HostUrl = "http://istio-gateway-istio.default.svc.cluster.local:80",
    [string]$LocustBaseUrl = "http://localhost:8089",
    [int]$SpawnRate = 5,
    [int]$SampleIntervalSeconds = 10,
    [string]$RunName = ("keda_real_test_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")),
    [ValidateSet("any", "keda", "hpa")]
    [string]$ScalerExpectation = "keda"
)

$ErrorActionPreference = "Stop"

# Same staged profile used in your historical tests.
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

$services = @(
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice"
)

$outDir = Join-Path "outputs\evaluations" $RunName
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$replicaRows = New-Object System.Collections.Generic.List[object]
$stageRows = New-Object System.Collections.Generic.List[object]

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

function Assert-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Get-ReplicaSnapshot {
    param(
        [Parameter(Mandatory = $true)][datetime]$Timestamp,
        [Parameter(Mandatory = $true)][int]$StageUsers,
        [Parameter(Mandatory = $true)][int]$StageDurationMinutes,
        [Parameter(Mandatory = $true)][int]$StageIndex
    )

    $deployJson = kubectl get deployment -n $Namespace -o json | ConvertFrom-Json
    $hpaJson = kubectl get hpa -n $Namespace -o json | ConvertFrom-Json

    $hpaByTarget = @{}
    foreach ($hpa in $hpaJson.items) {
        if ($hpa.spec.scaleTargetRef.kind -eq "Deployment") {
            $hpaByTarget[$hpa.spec.scaleTargetRef.name] = $hpa
        }
    }

    foreach ($service in $services) {
        $dep = $deployJson.items | Where-Object { $_.metadata.name -eq $service } | Select-Object -First 1
        $hpa = $hpaByTarget[$service]

        $replicaRows.Add([PSCustomObject]@{
            timestamp_utc = $Timestamp.ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss")
            run_name = $RunName
            namespace = $Namespace
            stage_index = $StageIndex
            stage_users = $StageUsers
            stage_duration_min = $StageDurationMinutes
            service = $service
            deployment_spec_replicas = if ($dep) { [int]$dep.spec.replicas } else { $null }
            deployment_status_replicas = if ($dep) { [int]$dep.status.replicas } else { $null }
            deployment_ready_replicas = if ($dep) { [int]$dep.status.readyReplicas } else { $null }
            deployment_available_replicas = if ($dep) { [int]$dep.status.availableReplicas } else { $null }
            deployment_updated_replicas = if ($dep) { [int]$dep.status.updatedReplicas } else { $null }
            hpa_name = if ($hpa) { $hpa.metadata.name } else { $null }
            hpa_current_replicas = if ($hpa) { [int]$hpa.status.currentReplicas } else { $null }
            hpa_desired_replicas = if ($hpa) { [int]$hpa.status.desiredReplicas } else { $null }
            hpa_min_replicas = if ($hpa) { [int]$hpa.spec.minReplicas } else { $null }
            hpa_max_replicas = if ($hpa) { [int]$hpa.spec.maxReplicas } else { $null }
        })
    }
}

Assert-CommandExists -Name "kubectl"

if ($ScalerExpectation -eq "keda") {
    Write-Host "Checking KEDA ScaledObjects in namespace '$Namespace'..." -ForegroundColor Cyan
    $scaledObjectsJson = kubectl get scaledobject -n $Namespace -o json | ConvertFrom-Json
    if (-not $scaledObjectsJson.items -or $scaledObjectsJson.items.Count -eq 0) {
        throw "No KEDA ScaledObjects found in namespace '$Namespace'. Enable/apply KEDA first."
    }

    $scaledTargets = $scaledObjectsJson.items | ForEach-Object { $_.spec.scaleTargetRef.name }
    $missingTargets = $services | Where-Object { $_ -notin $scaledTargets }
    if ($missingTargets.Count -gt 0) {
        Write-Warning "Some services have no ScaledObject: $($missingTargets -join ', ')"
    }
}
elseif ($ScalerExpectation -eq "hpa") {
    Write-Host "Checking HPAs in namespace '$Namespace'..." -ForegroundColor Cyan
    $hpaJson = kubectl get hpa -n $Namespace -o json | ConvertFrom-Json
    if (-not $hpaJson.items -or $hpaJson.items.Count -eq 0) {
        throw "No HPAs found in namespace '$Namespace'. Apply baseline autoscaling first."
    }

    $hpaTargets = $hpaJson.items | Where-Object { $_.spec.scaleTargetRef.kind -eq "Deployment" } | ForEach-Object { $_.spec.scaleTargetRef.name }
    $missingHpas = $services | Where-Object { $_ -notin $hpaTargets }
    if ($missingHpas.Count -gt 0) {
        Write-Warning "Some services have no HPA: $($missingHpas -join ', ')"
    }
}
else {
    Write-Host "Scaler expectation is 'any'; skipping scaler pre-check." -ForegroundColor Yellow
}

Write-Host "Starting KEDA validation run: $RunName" -ForegroundColor Green
Write-Host "Output folder: $outDir"

$runStart = Get-Date

try {
    for ($i = 0; $i -lt $stages.Count; $i++) {
        $stage = $stages[$i]
        $users = [int]$stage.Users
        $durationMinutes = [int]$stage.Duration
        $durationSeconds = $durationMinutes * 60
        $stageStart = Get-Date

        Write-Host "----------------------------------------"
        Write-Host "Stage $($i + 1)/$($stages.Count): users=$users duration=${durationMinutes}m"

        if ($users -eq 0) {
            Invoke-LocustApi -Uri "$LocustBaseUrl/stop" -Method GET | Out-Null
        }
        else {
            Invoke-LocustApi `
                -Uri "$LocustBaseUrl/swarm" `
                -Method POST `
                -Body @{
                    user_count = $users
                    spawn_rate = $SpawnRate
                    host = $HostUrl
                } | Out-Null
        }

        $elapsed = 0
        while ($elapsed -lt $durationSeconds) {
            $tickTs = Get-Date
            Get-ReplicaSnapshot `
                -Timestamp $tickTs `
                -StageUsers $users `
                -StageDurationMinutes $durationMinutes `
                -StageIndex ($i + 1)

            $sleepSeconds = [Math]::Min($SampleIntervalSeconds, $durationSeconds - $elapsed)
            Start-Sleep -Seconds $sleepSeconds
            $elapsed += $sleepSeconds
        }

        $stageEnd = Get-Date
        $stageRows.Add([PSCustomObject]@{
            run_name = $RunName
            stage_index = $i + 1
            users = $users
            duration_min = $durationMinutes
            started_at = $stageStart.ToString("yyyy-MM-dd HH:mm:ss")
            ended_at = $stageEnd.ToString("yyyy-MM-dd HH:mm:ss")
        })
    }
}
finally {
    try {
        Invoke-LocustApi -Uri "$LocustBaseUrl/stop" -Method GET | Out-Null
    }
    catch {
        Write-Warning "Could not stop Locust in finally block."
    }
}

$runEnd = Get-Date

$replicaCsv = Join-Path $outDir "keda_replica_timeline.csv"
$stageCsv = Join-Path $outDir "keda_stage_timeline.csv"
$summaryTxt = Join-Path $outDir "keda_run_summary.txt"

$replicaRows | Export-Csv -Path $replicaCsv -NoTypeInformation -Encoding UTF8
$stageRows | Export-Csv -Path $stageCsv -NoTypeInformation -Encoding UTF8

@(
    "run_name=$RunName"
    "namespace=$Namespace"
    "started_at=$($runStart.ToString('yyyy-MM-dd HH:mm:ss'))"
    "ended_at=$($runEnd.ToString('yyyy-MM-dd HH:mm:ss'))"
    "sample_interval_seconds=$SampleIntervalSeconds"
    "spawn_rate=$SpawnRate"
    "scaler_expectation=$ScalerExpectation"
    "records=$($replicaRows.Count)"
    "services=$($services -join ',')"
    "replica_csv=$replicaCsv"
    "stage_csv=$stageCsv"
) | Out-File -FilePath $summaryTxt -Force -Encoding utf8

Write-Host ""
Write-Host "KEDA validation finished." -ForegroundColor Green
Write-Host "Replica timeline: $replicaCsv"
Write-Host "Stage timeline:   $stageCsv"
Write-Host "Summary:          $summaryTxt"
