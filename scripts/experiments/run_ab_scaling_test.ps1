# A/B scaling test runner
# A: Proposed solution (KEDA + predicted replicas)
# B: Baseline autoscaling (CPU HPA)

param(
    [string]$Namespace = "default",
    [string]$ProposedManifestPath = "k8s/keda/all-services-predicted-replicas-scaledobjects.yaml",
    [string]$BaselineManifestPath = "k8s/hpa/all-services-cpu-hpa.yaml",
    [string]$HostUrl = "http://istio-gateway-istio.default.svc.cluster.local:80",
    [string]$LocustBaseUrl = "http://localhost:8089",
    [int]$SpawnRate = 5,
    [int]$SampleIntervalSeconds = 10,
    [int]$StabilizationSeconds = 90,
    [int]$ResetReplicas = 1,
    [string]$RunPrefix = ("ab_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
)

$ErrorActionPreference = "Stop"

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

function Assert-PathExists {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -Path $Path)) {
        throw "Required path not found: $Path"
    }
}

function Reset-ScalingObjects {
    Write-Host "Removing existing scaling objects for clean A/B setup..." -ForegroundColor Yellow

    foreach ($service in $services) {
        kubectl delete scaledobject "${service}-predicted-replicas-scaler" -n $Namespace --ignore-not-found | Out-Null
        kubectl delete hpa "${service}-cpu-hpa" -n $Namespace --ignore-not-found | Out-Null
    }

    # Safety cleanup for any lingering KEDA-generated HPAs.
    kubectl get hpa -n $Namespace -o name |
        Where-Object { $_ -like "*keda-hpa-*" } |
        ForEach-Object { kubectl delete $_ -n $Namespace --ignore-not-found | Out-Null }
}

function Reset-Deployments {
    param([Parameter(Mandatory = $true)][int]$Replicas)

    Write-Host "Resetting deployments to replicas=$Replicas..." -ForegroundColor Yellow
    foreach ($service in $services) {
        kubectl scale deployment/$service -n $Namespace --replicas=$Replicas | Out-Null
    }
}

function Run-OneVariant {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][ValidateSet("keda", "hpa", "any")][string]$ScalerExpectation
    )

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Starting variant: $Label"
    Write-Host "Manifest: $ManifestPath"
    Write-Host "Expectation: $ScalerExpectation"

    Reset-ScalingObjects

    Write-Host "Applying manifest: $ManifestPath" -ForegroundColor Cyan
    kubectl apply -f $ManifestPath | Out-Null

    Reset-Deployments -Replicas $ResetReplicas

    Write-Host "Waiting $StabilizationSeconds seconds for stabilization..." -ForegroundColor Cyan
    Start-Sleep -Seconds $StabilizationSeconds

    $variantRunName = "${RunPrefix}_${Label}"

    & ".\scripts\experiments\run_keda_replica_validation.ps1" `
        -Namespace $Namespace `
        -HostUrl $HostUrl `
        -LocustBaseUrl $LocustBaseUrl `
        -SpawnRate $SpawnRate `
        -SampleIntervalSeconds $SampleIntervalSeconds `
        -RunName $variantRunName `
        -ScalerExpectation $ScalerExpectation

    if ($LASTEXITCODE -ne 0) {
        throw "Variant '$Label' run failed with exit code $LASTEXITCODE"
    }

    return $variantRunName
}

Assert-PathExists -Path $ProposedManifestPath
Assert-PathExists -Path $BaselineManifestPath
Assert-PathExists -Path ".\scripts\experiments\run_keda_replica_validation.ps1"

$summaryDir = Join-Path "outputs\evaluations" $RunPrefix
New-Item -ItemType Directory -Path $summaryDir -Force | Out-Null

Write-Host "Running A/B scaling test with prefix: $RunPrefix" -ForegroundColor Green
Write-Host "A = proposed predicted KEDA"
Write-Host "B = baseline CPU HPA"

$runA = Run-OneVariant -Label "A_proposed" -ManifestPath $ProposedManifestPath -ScalerExpectation "keda"
$runB = Run-OneVariant -Label "B_baseline" -ManifestPath $BaselineManifestPath -ScalerExpectation "hpa"

$summaryFile = Join-Path $summaryDir "ab_test_summary.txt"
@(
    "run_prefix=$RunPrefix"
    "namespace=$Namespace"
    "proposed_manifest=$ProposedManifestPath"
    "baseline_manifest=$BaselineManifestPath"
    "run_a=$runA"
    "run_b=$runB"
    "a_replica_csv=outputs/evaluations/$runA/keda_replica_timeline.csv"
    "b_replica_csv=outputs/evaluations/$runB/keda_replica_timeline.csv"
    "a_stage_csv=outputs/evaluations/$runA/keda_stage_timeline.csv"
    "b_stage_csv=outputs/evaluations/$runB/keda_stage_timeline.csv"
) | Out-File -FilePath $summaryFile -Force -Encoding utf8

Write-Host ""
Write-Host "A/B test complete." -ForegroundColor Green
Write-Host "Summary: $summaryFile"
Write-Host "Run A: outputs/evaluations/$runA"
Write-Host "Run B: outputs/evaluations/$runB"
