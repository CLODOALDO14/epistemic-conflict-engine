<#
.SYNOPSIS
    Epistemic Conflict Engine - Tek Tik Yerel Baslatici
.DESCRIPTION
    Bu betik asagidaki adimlari otomatik olarak gerceklestirir:
      1. Python sanal ortamini (venv) olusturur / etkinlestirir
      2. pip ile gereksinimleri yukler
      3. Ollama servisini kontrol eder, gerekli modelleri ceker
      4. Neo4j Community Edition'i baslatir (yerel var/ dizininden)
      5. Demo korpusu yukler (bootstrap-demo --reset)
      6. Preflight kontrolu calistirir
      7. Interaktif menu sunar: start / resume / preflight / stop / exit
.NOTES
    PowerShell 5.1+ ve Windows 10/11 gerektirir.
    Sag tiklayip "PowerShell ile Calistir" ile baslatabilirsiniz.
#>

param(
    [switch]$SkipNeo4j,
    [switch]$SkipOllama,
    [switch]$SkipBootstrap,
    [switch]$NonInteractive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── Paths ────────────────────────────────────────────────────────────
$ProjectRoot   = $PSScriptRoot
$VenvDir       = Join-Path $ProjectRoot ".venv"
$VarDir        = Join-Path $ProjectRoot "var"
$EnvFile       = Join-Path $ProjectRoot ".env"
$ReqFile       = Join-Path $ProjectRoot "requirements.txt"
$AllInOne      = Join-Path $ProjectRoot "ECE_v3_ALL_IN_ONE.py"
$ExampleGround = Join-Path $ProjectRoot "examples" "material_grounding.example.json"

$JavaHome      = Join-Path $VarDir "java" "zulu21"
$Neo4jHome     = Join-Path $VarDir "neo4j" "neo4j-community-2026.03.1"
$Neo4jBin      = Join-Path $Neo4jHome "bin" "neo4j.bat"
$Neo4jConf     = Join-Path $Neo4jHome "conf" "neo4j.conf"

# ── Helpers ──────────────────────────────────────────────────────────
function Write-Banner {
    Write-Host ""
    Write-Host "  +====================================================+" -ForegroundColor Cyan
    Write-Host "  |   Epistemic Conflict Engine  v3.0                   |" -ForegroundColor Cyan
    Write-Host "  |   Tek Tik Yerel Baslatici                          |" -ForegroundColor Cyan
    Write-Host "  +====================================================+" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step ([string]$msg) {
    Write-Host "  >> $msg" -ForegroundColor Yellow
}

function Write-Ok ([string]$msg) {
    Write-Host "  [OK] $msg" -ForegroundColor Green
}

function Write-Err ([string]$msg) {
    Write-Host "  [FAIL] $msg" -ForegroundColor Red
}

function Write-Info ([string]$msg) {
    Write-Host "  [i] $msg" -ForegroundColor Gray
}

# Find Python 3.12 via .NET to avoid Turkish I path issues in PowerShell
function Find-Python {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe")
    )
    foreach ($c in $candidates) {
        if ([System.IO.File]::Exists($c)) {
            return $c
        }
    }
    # Fallback: try PATH
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    return $null
}

function Invoke-Python {
    param([string]$PythonExe, [string[]]$Arguments)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $PythonExe
    $psi.Arguments              = ($Arguments | ForEach-Object { "`"$_`"" }) -join " "
    $psi.WorkingDirectory       = $ProjectRoot
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError  = $false
    # Propagate env
    $psi.EnvironmentVariables["JAVA_HOME"] = $JavaHome
    $psi.EnvironmentVariables["NEO4J_HOME"] = $Neo4jHome
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()
    return $proc.ExitCode
}

function Invoke-PythonCapture {
    param([string]$PythonExe, [string[]]$Arguments)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $PythonExe
    $psi.Arguments              = ($Arguments | ForEach-Object { "`"$_`"" }) -join " "
    $psi.WorkingDirectory       = $ProjectRoot
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.EnvironmentVariables["JAVA_HOME"] = $JavaHome
    $psi.EnvironmentVariables["NEO4J_HOME"] = $Neo4jHome
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    return @{ ExitCode = $proc.ExitCode; StdOut = $stdout; StdErr = $stderr }
}

function Load-EnvFile {
    if (-not (Test-Path $EnvFile)) { return }
    foreach ($line in (Get-Content $EnvFile -Encoding UTF8)) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) { continue }
        $parts = $trimmed.Split("=", 2)
        $key   = $parts[0].Trim()
        $val   = $parts[1].Trim().Trim("'").Trim('"')
        if ($key -and -not [System.Environment]::GetEnvironmentVariable($key)) {
            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
}

# ── Step 1: Python venv ──────────────────────────────────────────────
function Setup-Venv {
    Write-Step "Python sanal ortami kontrol ediliyor..."

    $sysPython = Find-Python
    if (-not $sysPython) {
        Write-Err "Python bulunamadi! Python 3.11+ yuklu olmali."
        Write-Err "https://python.org/downloads adresinden yukleyip tekrar deneyin."
        return $null
    }
    Write-Info "Sistem Python: $sysPython"

    $venvPython = Join-Path $VenvDir "Scripts" "python.exe"
    $venvPip    = Join-Path $VenvDir "Scripts" "pip.exe"

    if (-not [System.IO.File]::Exists($venvPython)) {
        Write-Step "Sanal ortam olusturuluyor (.venv)..."
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $sysPython
        $psi.Arguments = "-m venv `"$VenvDir`""
        $psi.WorkingDirectory = $ProjectRoot
        $psi.UseShellExecute = $false
        $proc = [System.Diagnostics.Process]::Start($psi)
        $proc.WaitForExit()
        if ($proc.ExitCode -ne 0) {
            Write-Err "venv olusturulamadi (exit code: $($proc.ExitCode))."
            return $null
        }
    }
    Write-Ok "Sanal ortam hazir: $VenvDir"

    # Install requirements
    Write-Step "Pip bagimliliklari kontrol ediliyor..."
    $exitCode = Invoke-Python -PythonExe $venvPython -Arguments @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
    $exitCode = Invoke-Python -PythonExe $venvPython -Arguments @("-m", "pip", "install", "--quiet", "-r", $ReqFile)
    if ($exitCode -ne 0) {
        Write-Err "Pip install basarisiz (exit code: $exitCode). requirements.txt kontrol edin."
        return $null
    }
    Write-Ok "Tum Python bagimliliklari yuklu."
    return $venvPython
}

# ── Step 2: Ollama ───────────────────────────────────────────────────
function Setup-Ollama {
    Write-Step "Ollama kontrol ediliyor..."

    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        Write-Err "Ollama bulunamadi! https://ollama.com adresinden yukleyin."
        return $false
    }
    Write-Ok "Ollama bulundu: $($ollamaCmd.Source)"

    # Check if Ollama is serving
    Write-Step "Ollama servisi kontrol ediliyor..."
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
        Write-Ok "Ollama servisi calisiyor."
    }
    catch {
        Write-Step "Ollama servisi baslatiliyor..."
        Start-Process -FilePath $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10 -ErrorAction Stop | Out-Null
            Write-Ok "Ollama servisi baslatildi."
        }
        catch {
            Write-Err "Ollama servisi baslatilamadi. Manuel olarak 'ollama serve' calistirin."
            return $false
        }
    }

    # Pull required models
    Load-EnvFile
    $chatModel     = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "llama3.1" }
    $embedModel    = if ($env:OLLAMA_EMBEDDING_MODEL) { $env:OLLAMA_EMBEDDING_MODEL } else { "nomic-embed-text" }

    foreach ($model in @($chatModel, $embedModel)) {
        Write-Step "Model kontrol: $model ..."
        $pullProc = Start-Process -FilePath $ollamaCmd.Source -ArgumentList "pull",$model -NoNewWindow -Wait -PassThru
        if ($pullProc.ExitCode -eq 0) {
            Write-Ok "Model hazır: $model"
        }
        else {
            Write-Err "Model cekilemedi: $model (exit code: $($pullProc.ExitCode))"
            return $false
        }
    }
    return $true
}

# ── Step 3: Neo4j ────────────────────────────────────────────────────
function Setup-Neo4j {
    Write-Step "Neo4j kontrol ediliyor..."

    if (-not [System.IO.File]::Exists($Neo4jBin)) {
        Write-Err "Neo4j bulunamadı: $Neo4jBin"
        Write-Err "var/neo4j/neo4j-community-2026.03.1 dizininin mevcut oldugundan emin olun."
        return $false
    }

    # Set JAVA_HOME and NEO4J_HOME
    $env:JAVA_HOME  = $JavaHome
    $env:NEO4J_HOME = $Neo4jHome
    Write-Info "JAVA_HOME  = $JavaHome"
    Write-Info "NEO4J_HOME = $Neo4jHome"

    # Configure password (set initial password if needed)
    Load-EnvFile
    $neo4jPassword = if ($env:NEO4J_PASSWORD) { $env:NEO4J_PASSWORD } else { "ece-local-pass-2026" }

    # Check if Neo4j is already running
    $neo4jRunning = $false
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", 7687)
        $tcp.Close()
        $neo4jRunning = $true
    }
    catch { }

    if ($neo4jRunning) {
        Write-Ok "Neo4j zaten calisiyor (bolt://127.0.0.1:7687)."
        return $true
    }

    # Try setting initial password
    Write-Step "Neo4j ilk sifre ayarlaniyor..."
    $adminBat = Join-Path $Neo4jHome "bin" "neo4j-admin.bat"
    if ([System.IO.File]::Exists($adminBat)) {
        $setPwProc = Start-Process -FilePath $adminBat `
            -ArgumentList "dbms","set-initial-password",$neo4jPassword `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput "$env:TEMP\neo4j_setpw_out.txt" `
            -RedirectStandardError "$env:TEMP\neo4j_setpw_err.txt"
        # Ignore error if password already set
    }

    # Start Neo4j console in background
    Write-Step "Neo4j baslatiliyor..."
    $neo4jLogOut = Join-Path $VarDir "neo4j-console.out.log"
    $neo4jLogErr = Join-Path $VarDir "neo4j-console.err.log"

    $neo4jProc = Start-Process -FilePath $Neo4jBin -ArgumentList "console" `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $neo4jLogOut `
        -RedirectStandardError $neo4jLogErr

    # Store PID for later cleanup
    $script:Neo4jPID = $neo4jProc.Id
    Write-Info "Neo4j PID: $($neo4jProc.Id)"

    # Wait for bolt port
    Write-Step "Neo4j'nin hazir olmasi bekleniyor (max 90s)..."
    $deadline = (Get-Date).AddSeconds(90)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", 7687)
            $tcp.Close()
            $ready = $true
            break
        }
        catch { }
    }

    if ($ready) {
        Write-Ok "Neo4j hazir (bolt://127.0.0.1:7687)."
        return $true
    }
    else {
        Write-Err "Neo4j 90 saniye icinde baslatilamadi."
        Write-Info "Log: $neo4jLogErr"
        if (Test-Path $neo4jLogErr) {
            Get-Content $neo4jLogErr -Tail 10 | ForEach-Object { Write-Info "  $_" }
        }
        return $false
    }
}

# ── Step 4: Bootstrap demo ───────────────────────────────────────────
function Run-Bootstrap ([string]$PythonExe) {
    Write-Step "Demo korpus yukleniyor (bootstrap-demo --reset)..."
    $exitCode = Invoke-Python -PythonExe $PythonExe -Arguments @($AllInOne, "bootstrap-demo", "--reset")
    if ($exitCode -eq 0) {
        Write-Ok "Demo korpus yuklendi."
    }
    else {
        Write-Err "Bootstrap basarisiz (exit code: $exitCode)."
    }
    return $exitCode
}

# ── Step 5: Preflight ────────────────────────────────────────────────
function Run-Preflight ([string]$PythonExe) {
    Write-Step "Preflight kontrolleri calistiriliyor..."
    $exitCode = Invoke-Python -PythonExe $PythonExe -Arguments @($AllInOne, "preflight")
    if ($exitCode -eq 0) {
        Write-Ok "Preflight basarili - tum servisler hazir!"
    }
    else {
        Write-Err "Preflight basarisiz. Yukaridaki FAIL satirlarini kontrol edin."
    }
    return $exitCode
}

# ── Interactive Menu ─────────────────────────────────────────────────
function Show-Menu ([string]$PythonExe) {
    while ($true) {
        Write-Host ""
        Write-Host "  +--------------------------------------------+" -ForegroundColor Cyan
        Write-Host "  |  ECE Komut Menusu                          |" -ForegroundColor Cyan
        Write-Host "  +--------------------------------------------+" -ForegroundColor Cyan
        Write-Host "  |  1. start      - Yeni arastirma baslat     |" -ForegroundColor White
        Write-Host "  |  2. resume     - Duraklatilmis is devam    |" -ForegroundColor White
        Write-Host "  |  3. preflight  - Servis kontrolu           |" -ForegroundColor White
        Write-Host "  |  4. bootstrap  - Demo korpus yenile        |" -ForegroundColor White
        Write-Host "  |  5. stop       - Neo4j durdur ve cik       |" -ForegroundColor White
        Write-Host "  |  6. exit       - Cik (Neo4j calismaya      |" -ForegroundColor White
        Write-Host "  |                   devam eder)              |" -ForegroundColor White
        Write-Host "  +--------------------------------------------+" -ForegroundColor Cyan
        Write-Host ""
        $choice = Read-Host "  Seciminiz (1-6)"

        switch ($choice) {
            "1" {
                $topic    = Read-Host "  Konu (topic)"
                $year     = Read-Host "  Hedef yil (year)"
                $threadId = Read-Host "  Thread ID (ornek: prison-1975)"
                $rebuttal = Read-Host "  Rebuttal aktif mi? (e/h)"
                $eceArgs = @($AllInOne, "start", "--topic", $topic, "--year", $year, "--thread-id", $threadId)
                if ($rebuttal -eq "e") { $eceArgs += "--enable-rebuttal" }
                Write-Host ""
                Invoke-Python -PythonExe $PythonExe -Arguments $eceArgs | Out-Null
                Write-Host ""
                Write-Host "  --- Ipucu: Is duraklatildiysa 'resume' ile devam edin ---" -ForegroundColor DarkYellow
            }
            "2" {
                $threadId = Read-Host "  Thread ID"
                $groundingDefault = $ExampleGround
                $groundingFile = Read-Host "  Grounding JSON dosyasi (Enter = ornek dosya)"
                if (-not $groundingFile) { $groundingFile = $groundingDefault }
                Write-Host ""
                Invoke-Python -PythonExe $PythonExe -Arguments @($AllInOne, "resume", "--thread-id", $threadId, "--grounding-file", $groundingFile) | Out-Null
            }
            "3" {
                Run-Preflight -PythonExe $PythonExe | Out-Null
            }
            "4" {
                Run-Bootstrap -PythonExe $PythonExe | Out-Null
            }
            "5" {
                Write-Step "Neo4j durduruluyor..."
                if ($script:Neo4jPID) {
                    try {
                        Stop-Process -Id $script:Neo4jPID -Force -ErrorAction SilentlyContinue
                        Write-Ok "Neo4j durduruldu (PID: $($script:Neo4jPID))."
                    }
                    catch {
                        Write-Info "Neo4j zaten durmus olabilir."
                    }
                }
                else {
                    # Try stopping via neo4j stop
                    Start-Process -FilePath $Neo4jBin -ArgumentList "stop" -NoNewWindow -Wait -ErrorAction SilentlyContinue
                    Write-Ok "Neo4j stop komutu gonderildi."
                }
                Write-Host ""
                Write-Host "  Gule gule!" -ForegroundColor Cyan
                return
            }
            "6" {
                Write-Host ""
                Write-Info "Neo4j arka planda calismaya devam edecek."
                Write-Host "  Gule gule!" -ForegroundColor Cyan
                return
            }
            default {
                Write-Err "Gecersiz secim: $choice"
            }
        }
    }
}

# ======================================================================
#  MAIN
# ======================================================================
$script:Neo4jPID = $null

Write-Banner

# Load .env early
Load-EnvFile

# 1) Python venv
$VenvPython = Setup-Venv
if (-not $VenvPython) {
    Write-Err "Python kurulumu basarisiz. Cikiliyor."
    Read-Host "Devam etmek icin Enter'a basin"
    exit 1
}

# 2) Ollama
if (-not $SkipOllama) {
    $ollamaOk = Setup-Ollama
    if (-not $ollamaOk) {
        Write-Err "Ollama kurulumu basarisiz. -SkipOllama ile atlayabilirsiniz."
        $continue = Read-Host "Yine de devam etmek istiyor musunuz? (e/h)"
        if ($continue -ne "e") { exit 1 }
    }
}
else {
    Write-Info "Ollama adimi atlandi (-SkipOllama)."
}

# 3) Neo4j
if (-not $SkipNeo4j) {
    $neo4jOk = Setup-Neo4j
    if (-not $neo4jOk) {
        Write-Err "Neo4j baslatilamadi. -SkipNeo4j ile atlayabilirsiniz."
        $continue = Read-Host "Yine de devam etmek istiyor musunuz? (e/h)"
        if ($continue -ne "e") { exit 1 }
    }
}
else {
    Write-Info "Neo4j adimi atlandi (-SkipNeo4j)."
}

# 4) Bootstrap demo
if (-not $SkipBootstrap) {
    Run-Bootstrap -PythonExe $VenvPython | Out-Null
}
else {
    Write-Info "Bootstrap adimi atlandi (-SkipBootstrap)."
}

# 5) Preflight
Run-Preflight -PythonExe $VenvPython | Out-Null

# 6) Interactive menu or exit
if ($NonInteractive) {
    Write-Ok "NonInteractive mod - kurulum tamamlandi."
    exit 0
}

Show-Menu -PythonExe $VenvPython
