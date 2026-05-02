@echo off
setlocal

echo ==========================================
echo   CLEAN STOP - Online Boutique Stack
echo ==========================================
echo.

REM ===== Config =====
set PROJECT_ROOT=%~dp0..\..
set K8S_DIR=%PROJECT_ROOT%\k8s

set APP_MANIFEST=%K8S_DIR%\app\kubernetes-manifests.yaml
set ISTIO_MANIFEST=%K8S_DIR%\istio\istio-manifests.yaml
set LOCUST_SCRIPT=%K8S_DIR%\locust\locust-script.yaml
set LOCUST_TEST=%K8S_DIR%\locust\locust-test.yaml
set LOCUST_MOUNT=%K8S_DIR%\locust\locust-workload-pv-pvc.yaml
set ISTIO_PODMONITOR=%K8S_DIR%\istio\istio-podmonitor.yaml
set PREDICTED_EXPORTER=%K8S_DIR%\prometheus\predicted-metrics-exporter.yaml
set PREDICTED_EXPORTER_MONITOR=%K8S_DIR%\prometheus\predicted-metrics-servicemonitor.yaml
set KEDA_SCALEDOBJECT=%K8S_DIR%\keda\frontend-predicted-rps-scaledobject.yaml

REM =========================================================
REM 1. Remove Locust test resources first
REM =========================================================
echo [1/7] Removing Locust test resources...

kubectl delete locusttest online-boutique-test -n default --ignore-not-found 2>nul

if exist "%LOCUST_TEST%" (
    kubectl delete -f "%LOCUST_TEST%" --ignore-not-found
)

if exist "%LOCUST_SCRIPT%" (
    kubectl delete -f "%LOCUST_SCRIPT%" --ignore-not-found
)

kubectl delete job online-boutique-test-master -n default --ignore-not-found 2>nul
kubectl delete job online-boutique-test-worker -n default --ignore-not-found 2>nul
kubectl delete svc online-boutique-test-master -n default --ignore-not-found 2>nul
kubectl delete svc online-boutique-test-webui -n default --ignore-not-found 2>nul

REM =========================================================
REM 2. Delete Online Boutique app resources
REM =========================================================
echo.

echo Removing KEDA ScaledObject...
if exist "%KEDA_SCALEDOBJECT%" (
    kubectl delete -f "%KEDA_SCALEDOBJECT%" --ignore-not-found
)

echo [2/7] Deleting Online Boutique app...
if exist "%APP_MANIFEST%" (
    kubectl delete -f "%APP_MANIFEST%" --ignore-not-found
) else (
    echo WARNING: %APP_MANIFEST% not found. Skipping.
)

REM =========================================================
REM 3. Delete Istio application resources
REM =========================================================
echo.
echo [3/7] Removing Istio application resources...
if exist "%ISTIO_MANIFEST%" (
    kubectl delete -f "%ISTIO_MANIFEST%" --ignore-not-found
) else (
    echo WARNING: %ISTIO_MANIFEST% not found. Skipping.
)

kubectl label namespace default istio-injection- 2>nul

REM =========================================================
REM 4. Remove Locust Operator
REM =========================================================
echo.
echo [4/7] Removing Locust Operator...
helm uninstall locust-operator -n locust-operator 2>nul
kubectl delete ns locust-operator --ignore-not-found

REM =========================================================
REM 5. Remove Prometheus stack and Istio PodMonitor
REM =========================================================
echo.
echo [5/7] Removing Prometheus stack...

echo Removing predicted metrics ServiceMonitor...
if exist "%PREDICTED_EXPORTER_MONITOR%" (
    kubectl delete -f "%PREDICTED_EXPORTER_MONITOR%" --ignore-not-found
)

echo Removing predicted metrics exporter...
if exist "%PREDICTED_EXPORTER%" (
    kubectl delete -f "%PREDICTED_EXPORTER%" --ignore-not-found
)

kubectl delete podmonitor istio-sidecars -n monitoring --ignore-not-found 2>nul


helm uninstall kube-prometheus-stack -n monitoring 2>nul
kubectl delete ns monitoring --ignore-not-found


REM =========================================================
REM 5.1 Remove KEDA
REM =========================================================
echo.
echo Removing KEDA...
helm uninstall keda -n keda 2>nul
kubectl delete ns keda --ignore-not-found


REM =========================================================
REM 6. Remove Istio control plane
REM =========================================================
echo.
echo [6/7] Removing Istio control plane...
helm uninstall istiod -n istio-system 2>nul
helm uninstall istio-base -n istio-system 2>nul
kubectl delete ns istio-system --ignore-not-found


REM =========================================================
REM 7. Standard cleanup complete
REM =========================================================
echo.
echo [7/7] Standard cleanup complete.
echo.
echo NOTE:
echo - This script does NOT delete CRDs.
echo - Use Docker Desktop Reset Kubernetes only when you want a full cluster reset.

echo.
echo ==========================================
echo Cleanup completed.
echo ==========================================
echo.

pause
endlocal