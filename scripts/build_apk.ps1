<#
.SYNOPSIS
  构建「DSH 构图相机」v1 的 debug APK。
  只使用本机已装好的 JDK + Android SDK + gradle wrapper，不下载任何东西。

.DESCRIPTION
  1) 校验 -AndroidSdk / -JdkHome，设置 JAVA_HOME / ANDROID_HOME
  2) 在工程根目录写 local.properties（sdk.dir=...）
  3) 用 gradlew.bat（没有则用 PATH 里的 gradle）执行 assembleDebug
  4) 失败打印最后 40 行日志并 exit 1；成功把 APK 复制到 -OutApk 并打印绝对路径 + 大小(MB)

.EXAMPLE
  pwsh -File .\scripts\build_apk.ps1 -AndroidSdk "C:\Android\Sdk" -JdkHome "C:\Program Files\Eclipse Adoptium\jdk-17"

.NOTES
  本文件保存为 UTF-8（带 BOM），Windows PowerShell 5.1 与 PowerShell 7 都能正确显示中文。
#>
[CmdletBinding()]
param(
    [string]$AndroidSdk = $env:ANDROID_HOME,
    [string]$JdkHome    = $env:JAVA_HOME,
    [string]$OutApk     = "",
    [string]$ProjectDir = ""
)

$ErrorActionPreference = "Stop"

function Fail([string]$msg) {
    Write-Host ""
    Write-Host "[失败] $msg" -ForegroundColor Red
    exit 1
}

# ---------- 0. 工程根 = scripts 的上一级 ----------
if ([string]::IsNullOrWhiteSpace($ProjectDir)) { $ProjectDir = Split-Path -Parent $PSScriptRoot }
try { $ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path }
catch { Fail "工程目录不存在：$ProjectDir" }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir "settings.gradle.kts")) -and
    -not (Test-Path -LiteralPath (Join-Path $ProjectDir "settings.gradle"))) {
    Write-Host "[警告] $ProjectDir 下没看到 settings.gradle(.kts)，确认这是 Android 工程根目录。" -ForegroundColor Yellow
}

# ---------- 0.1 默认输出位置 = 工程根的 dist\ ----------
if ([string]::IsNullOrWhiteSpace($OutApk)) { $OutApk = Join-Path $ProjectDir "dist\dshcam-v1.apk" }

# ---------- 1. 定位 Android SDK / JDK ----------
if ([string]::IsNullOrWhiteSpace($AndroidSdk)) {
    Fail "没有拿到 Android SDK 路径。请加参数：-AndroidSdk `"C:\Android\Sdk`"（或先设置环境变量 ANDROID_HOME）"
}
if (-not (Test-Path -LiteralPath $AndroidSdk)) { Fail "Android SDK 目录不存在：$AndroidSdk" }
$AndroidSdk = (Resolve-Path -LiteralPath $AndroidSdk).Path

$javaExe = ""
if (-not [string]::IsNullOrWhiteSpace($JdkHome)) {
    $cand = Join-Path $JdkHome "bin\java.exe"
    if (Test-Path -LiteralPath $cand) {
        $JdkHome = (Resolve-Path -LiteralPath $JdkHome).Path
        $javaExe = Join-Path $JdkHome "bin\java.exe"
    } else {
        Fail "JDK 目录里找不到 bin\java.exe：$JdkHome（需要 JDK 17，不是 JRE）"
    }
} else {
    $cmd = Get-Command java -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Fail "没找到 JDK。请加参数：-JdkHome `"C:\Program Files\Eclipse Adoptium\jdk-17`"（或先设置环境变量 JAVA_HOME）"
    }
    $javaExe = $cmd.Source
    $JdkHome = Split-Path -Parent (Split-Path -Parent $javaExe)
    Write-Host "[提示] 未传 -JdkHome，改用 PATH 里的 java：$javaExe" -ForegroundColor Yellow
}

$env:JAVA_HOME = $JdkHome
$env:ANDROID_HOME = $AndroidSdk
$env:ANDROID_SDK_ROOT = $AndroidSdk
$env:Path = (Join-Path $JdkHome "bin") + ";" + (Join-Path $AndroidSdk "platform-tools") + ";" + $env:Path
Write-Host "[环境] JAVA_HOME   = $JdkHome"
Write-Host "[环境] ANDROID_HOME = $AndroidSdk"

# ---------- 2. 写 local.properties ----------
$localProps = Join-Path $ProjectDir "local.properties"
$escaped = $AndroidSdk.Replace('\', '\\').Replace(':', '\:')
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($localProps, "sdk.dir=$escaped`r`n", $utf8NoBom)
Write-Host "[1/3] 已写 $localProps  (sdk.dir=$escaped)"

# ---------- 3. 选择 gradle ----------
$gradleCmd = ""
$gradleArgs = @("assembleDebug", "--console=plain", "--stacktrace")
$gradlew = Join-Path $ProjectDir "gradlew.bat"
if (Test-Path -LiteralPath $gradlew) {
    $gradleCmd = $gradlew
} else {
    $g = Get-Command gradle -ErrorAction SilentlyContinue
    if ($null -eq $g) {
        Fail "工程里没有 gradlew.bat，PATH 里也没有 gradle 命令。请先用 Android Studio 打开工程生成 gradle wrapper。"
    }
    $gradleCmd = $g.Source
    Write-Host "[提示] 没找到 gradlew.bat，改用 PATH 里的 gradle：$gradleCmd" -ForegroundColor Yellow
}

$logDir = Join-Path $ProjectDir "build"
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$logFile = Join-Path $logDir "build_apk.log"

Write-Host "[2/3] 开始构建（日志：$logFile）"
Write-Host "      $gradleCmd $($gradleArgs -join ' ')"
$exitCode = 1
Push-Location $ProjectDir
try {
    & $gradleCmd @gradleArgs 2>&1 | Tee-Object -FilePath $logFile
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "===== 构建失败（exit $exitCode），最后 40 行日志 =====" -ForegroundColor Red
    if (Test-Path -LiteralPath $logFile) {
        Get-Content -LiteralPath $logFile -Tail 40 | ForEach-Object { Write-Host $_ }
    }
    Write-Host "完整日志：$logFile" -ForegroundColor Red
    exit 1
}

# ---------- 4. 收集 APK ----------
$apkSrc = Join-Path $ProjectDir "app\build\outputs\apk\debug\app-debug.apk"
if (-not (Test-Path -LiteralPath $apkSrc)) {
    $apkRoot = Join-Path $ProjectDir "app\build\outputs\apk"
    $found = $null
    if (Test-Path -LiteralPath $apkRoot) {
        $found = Get-ChildItem -Path $apkRoot -Recurse -Filter *.apk -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    if ($null -eq $found) { Fail "gradle 返回成功，但没找到 APK 产物（预期：$apkSrc）。请检查 app/build.gradle 的 applicationVariants。" }
    $apkSrc = $found.FullName
}

$outDir = Split-Path -Parent $OutApk
if (-not [string]::IsNullOrWhiteSpace($outDir) -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
Copy-Item -LiteralPath $apkSrc -Destination $OutApk -Force

$fullOut = (Resolve-Path -LiteralPath $OutApk).Path
$sizeMb = [math]::Round((Get-Item -LiteralPath $fullOut).Length / 1MB, 2)
Write-Host "[3/3] 构建完成" -ForegroundColor Green
Write-Host ""
Write-Host "[成功] APK 绝对路径：$fullOut" -ForegroundColor Green
Write-Host ("[成功] APK 大小：{0} MB" -f $sizeMb) -ForegroundColor Green
Write-Host ""
Write-Host "下一步（连上手机后安装）："
Write-Host "  pwsh -File `"$(Join-Path $PSScriptRoot 'install_apk.ps1')`" -Apk `"$fullOut`" -AndroidSdk `"$AndroidSdk`""
