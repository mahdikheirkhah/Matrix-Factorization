# Configuration
$ENV_NAME = "ex00"
$REQ_FILE = "requirements.txt"
$PORT = 8891

# --- Helper Functions ---

function Stop-Jupyter {
    Write-Host "Checking for Jupyter server on port $PORT..." -ForegroundColor Cyan
    $process = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue | 
               Select-Object -ExpandProperty OwningProcess -Unique
    if ($process) {
        Write-Host "Stopping Jupyter server (PID: $process)..." -ForegroundColor Yellow
        Stop-Process -Id $process -Force
    }
}

function Cleanup-Kernels {
    Write-Host "Scrubbing stale kernels..." -ForegroundColor Cyan
    jupyter kernelspec uninstall $ENV_NAME -f 2>$null
    # Windows-specific: cleanup leftover localhost kernels if they exist
    $kernels = jupyter kernelspec list | Select-String "localhost|127.0.0.1"
    foreach ($k in $kernels) {
        $name = ($k -split '\s+')[0]
        if ($name) { jupyter kernelspec uninstall $name -f 2>$null }
    }
}

function Set-VSCodeSettings {
    param($Token)
    Write-Host "Configuring VS Code workspace settings..." -ForegroundColor Cyan
    if (!(Test-Path .vscode)) { New-Item -ItemType Directory .vscode }
    $AbsPath = Get-Location
    $Settings = @{
        "python.defaultInterpreterPath" = "$AbsPath\$ENV_NAME\Scripts\python.exe"
        "jupyter.jupyterServerType" = "local"
        "jupyter.notebookEditor.defaultKernel" = ".jupyter/kernels/$ENV_NAME"
        "python.terminal.activateEnvInSelectedTerminal" = $true
        "jupyter.jupyterServerEndpoint" = "http://localhost:$PORT/?token=$Token"
    }
    $Settings | ConvertTo-Json | Out-File .vscode\settings.json -Encoding utf8
}

function Ensure-JupyterInstalled {
    Write-Host "Checking Jupyter installation..."
    $pipPath = "$ENV_NAME\Scripts\pip.exe"
    & $pipPath show notebook > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Jupyter Notebook not found. Installing..." -ForegroundColor Yellow
        & $pipPath install jupyter
        if (!(Test-Path $REQ_FILE)) { New-Item $REQ_FILE -ItemType File }
        # Simple check/append to requirements
        $content = Get-Content $REQ_FILE
        if ($content -notmatch "jupyter") { "jupyter" | Add-Content $REQ_FILE }
        if ($content -notmatch "notebook") { "notebook" | Add-Content $REQ_FILE }
        if ($content -notmatch "ipykernel") { "ipykernel" | Add-Content $REQ_FILE }
    }
}

function Launch-TerminalAndRun {
    Write-Host "Opening Jupyter in a new window..." -ForegroundColor Cyan
    $Cmd = "cd '$PWD'; .\$ENV_NAME\Scripts\activate; jupyter notebook --port $PORT --no-browser"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $Cmd
    
    Write-Host "Waiting for Jupyter to start..."
    Start-Sleep -Seconds 6
    
    $ServerInfo = jupyter server list | Select-String "token=([^ ]+)"
    if ($ServerInfo) {
        $Token = $ServerInfo.Matches.Groups[1].Value
        Set-VSCodeSettings $Token
        Write-Host "Token found and VS Code configured." -ForegroundColor Green
    }
}

function Create-And-Init {
    if (Test-Path $ENV_NAME) {
        Write-Host "Environment exists. Activating..." -ForegroundColor Green
    } else {
        Write-Host "Creating $ENV_NAME..." -ForegroundColor Cyan
        python -m venv $ENV_NAME
        if (!(Test-Path .gitignore)) { New-Item .gitignore -ItemType File }
        foreach ($entry in @($ENV_NAME, ".DS_Store")) {
            if ((Get-Content .gitignore) -notcontains $entry) { $entry | Add-Content .gitignore }
        }
    }

    # Activation for current session
    & ".\$ENV_NAME\Scripts\Activate.ps1"
    
    $pip = ".\$ENV_NAME\Scripts\pip.exe"
    & $pip install --upgrade pip
    & $pip install numpy pandas matplotlib scikit-learn
    
    if ((Get-Item $REQ_FILE).Length -gt 0) {
        & $pip install -r $REQ_FILE
    }

    Ensure-JupyterInstalled
    & ".\$ENV_NAME\Scripts\python.exe" -m ipykernel install --user --name="$ENV_NAME" --display-name="Python ($ENV_NAME)"
    
    Stop-Jupyter
    Cleanup-Kernels
    Launch-TerminalAndRun
    
    & $pip freeze > $REQ_FILE
    Write-Host "✅ Setup finished." -ForegroundColor Green
}

# --- Logic Flow ---

if ($args.Count -eq 0) {
    Create-And-Init
} else {
    switch ($args[0]) {
        "-c" { python -m venv $ENV_NAME }
        "-a" { & ".\$ENV_NAME\Scripts\Activate.ps1" }
        "-i" { 
            $pkgs = $args[1..($args.Count-1)]
            & ".\$ENV_NAME\Scripts\pip.exe" install $pkgs
            & ".\$ENV_NAME\Scripts\pip.exe" freeze > $REQ_FILE
        }
        "--run" { Launch-TerminalAndRun }
        "-d" { if ($env:VIRTUAL_ENV) { deactivate } }
        "--delete" {
            Stop-Jupyter
            Cleanup-Kernels
            if ($env:VIRTUAL_ENV) { deactivate }
            Remove-Item -Recurse -Force $ENV_NAME, .vscode\settings.json -ErrorAction SilentlyContinue
            Write-Host "Deleted everything." -ForegroundColor Red
        }
        Default { Write-Host "Unknown flag: $($args[0])" }
    }
}