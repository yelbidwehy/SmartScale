
Write-Host "======================================"
Write-Host "Merging all runs into one dataset..."
Write-Host "======================================"

python .\scripts\data_pcollection\merge_runs.py

Write-Host "Dataset ready: data\processed\smartscale_training_dataset.csv"