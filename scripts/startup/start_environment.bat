@echo off
setlocal enabledelayedexpansion

echo =========================================================
echo Online Boutique + Istio + Prometheus + Grafana + Locust
echo =========================================================
echo.

for %%i in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fi"
set K8S_DIR=%PROJECT_ROOT%\k8s
set DATA_DIR=%PROJECT_ROOT%\data

set APP_MANIFEST=%K8S_DIR%\app\kubernetes-manifests.yaml
set ISTIO_MANIFEST=%K8S_DIR%\istio\istio-manifests.yaml
set ISTIO_PODMONITOR=%K8S_DIR%\istio\istio-podmonitor.yaml

set LOCUST_SCRIPT=%K8S_DIR%\locust\locust-script.yaml
set LOCUST_TEST=%K8S_DIR%\locust\locust-test.yaml
set LOCUST_MOUNT=%K8S_DIR%\locust\locust-workload-pv-pvc.yaml
set LOCUST_PY=%K8S_DIR%\locust\locustfile.py

set WORKLOAD_JSON=%DATA_DIR%\raw\workload\boutique_workload_large.json

set KEDA_SCALEDOBJECT=%K8S_DIR%\keda\all-services-predicted-replicas-scaledobjects.yaml

set EXPORTER_NAME=predicted-metrics-exporter
set EXPORTER_IMAGE=predicted-metrics-exporter:latest

set EXPORTER_DOCKERFILE=%PROJECT_ROOT%\monitoring\Dockerfile
set "EXPORTER_TAR=%PROJECT_ROOT%\monitoring\predicted-metrics-exporter.tar"

set PREDICTED_EXPORTER=%K8S_DIR%\prometheus\predicted-metrics-exporter.yaml
set PREDICTED_EXPORTER_MONITOR=%K8S_DIR%\prometheus\predicted-metrics-servicemonitor.yaml



REM =========================================================
REM 0. Check required tools
REM =========================================================
echo [0/12] Checking required tools...
where kubectl >nul 2>nul || (echo ERROR: kubectl not found & exit /b 1)
where helm >nul 2>nul || (echo ERROR: helm not found & exit /b 1)
where git >nul 2>nul || (echo ERROR: git not found & exit /b 1)

REM =========================================================
REM 1. Check Kubernetes cluster
REM =========================================================
echo [1/12] Checking Kubernetes cluster...
kubectl get nodes >nul 2>nul || (echo ERROR: Kubernetes cluster is not reachable & exit /b 1)


REM =========================================================
REM 3. Install Gateway API CRDs
REM =========================================================
echo [3/12] Installing Gateway API CRDs...
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/latest/download/standard-install.yaml || exit /b 1

REM =========================================================
REM 4. Add / update Helm repos
REM =========================================================
echo [4/12] Adding Helm repos...
helm repo add istio https://istio-release.storage.googleapis.com/charts >nul 2>nul
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >nul 2>nul
helm repo add locust-operator https://charts.deliveryhero.io/ >nul 2>nul
helm repo add kedacore https://kedacore.github.io/charts >nul 2>nul
helm repo update || exit /b 1

REM =========================================================
REM 5. Install Prometheus stack
REM =========================================================
echo [5/12] Installing Prometheus stack...
kubectl get ns monitoring >nul 2>nul || kubectl create ns monitoring
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack -n monitoring ^
  --set grafana.adminUser=admin ^
  --set grafana.adminPassword=P@ssw0rd ^
  -n monitoring || exit /b 1

echo Waiting for monitoring operator...
kubectl rollout status deployment/kube-prometheus-stack-operator -n monitoring --timeout=600s || exit /b 1

echo Waiting for Grafana...
kubectl rollout status deployment/kube-prometheus-stack-grafana -n monitoring --timeout=600s || exit /b 1

echo Waiting for Prometheus pod...
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=600s || exit /b 1

echo Waiting for Alertmanager pod...
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=alertmanager -n monitoring --timeout=600s || exit /b 1





REM =========================================================
REM 5.2 Add ServiceMonitor for Predicted Metrics Exporter
REM =========================================================
echo [5.2/12] Adding predicted metrics ServiceMonitor...
kubectl apply -f "%PREDICTED_EXPORTER_MONITOR%" || exit /b 1


REM =========================================================
REM 5.3 Install KEDA
REM =========================================================
echo [5.3/12] Installing KEDA...
kubectl get ns keda >nul 2>nul || kubectl create ns keda
helm upgrade --install keda kedacore/keda -n keda || exit /b 1

echo Waiting for KEDA operator...
kubectl rollout status deployment/keda-operator -n keda --timeout=600s || exit /b 1

echo Waiting for KEDA metrics API server...
kubectl rollout status deployment/keda-operator-metrics-apiserver -n keda --timeout=600s || exit /b 1

echo Waiting for KEDA admission webhooks...
kubectl rollout status deployment/keda-admission-webhooks -n keda --timeout=600s || exit /b 1

REM =========================================================
REM 6. Install Istio
REM =========================================================
echo [6/12] Installing Istio...
kubectl get ns istio-system >nul 2>nul || kubectl create ns istio-system
helm upgrade --install istio-base istio/base -n istio-system || exit /b 1
helm upgrade --install istiod istio/istiod -n istio-system || exit /b 1
kubectl rollout status deployment/istiod -n istio-system --timeout=600s || exit /b 1

REM =========================================================
REM 7. Enable sidecar injection
REM =========================================================
echo [7/12] Enabling Istio sidecar injection on default namespace...
kubectl label namespace default istio-injection=enabled --overwrite || exit /b 1

REM =========================================================
REM 8. Deploy Online Boutique
REM =========================================================
echo [8/12] Deploying Online Boutique...
kubectl apply -f "%APP_MANIFEST%" || exit /b 1

echo Restarting deployments in default namespace to inject sidecars...
kubectl rollout restart deployment -n default || exit /b 1

echo Waiting for frontend deployment...
kubectl rollout status deployment/frontend -n default --timeout=600s || exit /b 1

REM =========================================================
REM 8.1 Apply KEDA ScaledObject
REM =========================================================
echo [8.1/12] Applying KEDA ScaledObject...
kubectl apply -f "%KEDA_SCALEDOBJECT%" || exit /b 1

echo Checking KEDA ScaledObject...
kubectl get scaledobject frontend-predicted-rps-scaler -n default
kubectl get hpa

REM =========================================================
REM 9. Apply Istio manifests
REM =========================================================
echo [9/12] Applying Istio manifests...
kubectl apply -f "%ISTIO_MANIFEST%" || exit /b 1

REM =========================================================
REM 10. Add Istio PodMonitor for Prometheus
REM =========================================================

kubectl apply -f "%ISTIO_PODMONITOR%" || exit /b 1

echo Restarting Prometheus to pick up PodMonitor...
kubectl rollout restart statefulset prometheus-kube-prometheus-stack-prometheus -n monitoring || exit /b 1
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=600s || exit /b 1

REM =========================================================
REM 11. Install Locust Operator
REM =========================================================
echo [11/12] Installing Locust Operator...
kubectl get ns locust-operator >nul 2>nul || kubectl create ns locust-operator
helm upgrade --install locust-operator locust-operator/locust-operator -n locust-operator || exit /b 1

REM =========================================================
REM 12. Apply Locust files and wait
REM =========================================================
echo [12/12] Applying Locust files...


    kubectl apply -f "%LOCUST_SCRIPT%" 

    kubectl apply -f "%LOCUST_TEST%" 
    


echo Waiting for Locust master pod...
kubectl wait --for=condition=Ready pod -l job-name=online-boutique-test-master -n default --timeout=600s

echo Waiting for Locust worker pods...
kubectl wait --for=condition=Ready pod -l job-name=online-boutique-test-worker -n default --timeout=600s

echo Waiting for Locust Web UI service...
:wait_locust_svc
kubectl get svc online-boutique-test-webui -n default >nul 2>nul
if errorlevel 1 (
    timeout /t 5 >nul
    goto wait_locust_svc
)



echo.
echo =========================================================
echo Setup completed successfully
echo =========================================================
echo.

echo Starting port-forwards...

wt ^
new-tab -p "Command Prompt" cmd /k "title Frontend && kubectl port-forward svc/frontend-external 8080:80" ^
; new-tab -p "Command Prompt" cmd /k "title Prometheus && kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090" ^
; new-tab -p "Command Prompt" cmd /k "title Grafana && kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80" ^
; new-tab -p "Command Prompt" cmd /k "title Locust && kubectl port-forward -n default svc/online-boutique-test-webui 8089:8089" ^
; new-tab -p "Command Prompt" cmd /k "title Predicted Metrics && kubectl port-forward -n default svc/predicted-metrics-exporter 8000:8000"

echo.
echo =========================================================
echo URLs
echo =========================================================
echo Online Boutique: http://localhost:8080
echo Prometheus:      http://localhost:9090
echo Grafana:         http://localhost:3000
echo Locust:          http://localhost:8089
echo Predicted Metrics: http://localhost:8000/metrics
echo.

echo Grafana username: admin
echo Grafana password in PowerShell:
echo kubectl get secret -n monitoring kube-prometheus-stack-grafana -o jsonpath="{.data.admin-password}" ^| %%{ [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }

echo Grafana username: admin
echo Grafana password: P@ssw0rd

echo.
pause
endlocal