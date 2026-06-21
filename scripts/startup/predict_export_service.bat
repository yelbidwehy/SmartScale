
@echo off
setlocal EnableExtensions

if not defined PROJECT_ROOT (
    for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
)

if not defined EXPORTER_NAME set "EXPORTER_NAME=predicted-metrics-exporter"
if not defined EXPORTER_IMAGE set "EXPORTER_IMAGE=predicted-metrics-exporter:latest"
if not defined PREDICTED_EXPORTER set "PREDICTED_EXPORTER=%PROJECT_ROOT%\k8s\prometheus\predicted-metrics-exporter.yaml"
if not defined PREDICTED_EXPORTER_MONITOR set "PREDICTED_EXPORTER_MONITOR=%PROJECT_ROOT%\k8s\prometheus\predicted-metrics-servicemonitor.yaml"

REM =========================================================
REM 5.1 Deploy Predicted Metrics Exporter
REM =========================================================
echo [5.1/12] Deploying predicted metrics exporter...

echo [%EXPORTER_NAME%] Building Docker image...

set EXPORTER_DOCKERFILE=%PROJECT_ROOT%\monitoring\Dockerfile
set "EXPORTER_TAR=%PROJECT_ROOT%\monitoring\predicted-metrics-exporter.tar"
set "EXPORTER_TAR_NAME=predicted-metrics-exporter.tar"

docker build -t "%EXPORTER_IMAGE%" -f "%EXPORTER_DOCKERFILE%" "%PROJECT_ROOT%"
if errorlevel 1 exit /b 1

echo [%EXPORTER_NAME%] Saving image...

docker save -o "%EXPORTER_TAR%" "%EXPORTER_IMAGE%"
if errorlevel 1 exit /b 1

echo [%EXPORTER_NAME%] Importing image into Kubernetes nodes...

set "IMPORT_COUNT=0"

for /f "tokens=*" %%N in ('docker ps --format "{{.Names}}" ^| findstr /R "^desktop-control-plane$ ^desktop-worker.*$"') do (
    echo Importing into %%N...
    set /a IMPORT_COUNT+=1

    docker cp "%EXPORTER_TAR%" %%N:/%EXPORTER_TAR_NAME%
    if errorlevel 1 exit /b 1

    docker exec %%N ctr -n k8s.io images import /%EXPORTER_TAR_NAME%
    if errorlevel 1 exit /b 1
)

if "%IMPORT_COUNT%"=="0" (
    echo [%EXPORTER_NAME%] No visible docker-desktop node containers were found.
    echo [%EXPORTER_NAME%] Continuing without manual image import.
)

echo [%EXPORTER_NAME%] Applying Kubernetes manifests...

kubectl apply -f "%PREDICTED_EXPORTER%"
if errorlevel 1 exit /b 1

kubectl apply -f "%PREDICTED_EXPORTER_MONITOR%"
if errorlevel 1 exit /b 1

echo [%EXPORTER_NAME%] Waiting for pod...

kubectl wait --for=condition=Ready pod -l app=%EXPORTER_NAME% -n monitoring --timeout=300s
if errorlevel 1 exit /b 1

echo [%EXPORTER_NAME%] Ready.
