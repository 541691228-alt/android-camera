<#
.SYNOPSIS
  把 dshcam v1 的 APK 装到通过 USB 连接的小米 14 Ultra 上（adb install -r）。

.DESCRIPTION
  -Apk  APK 文件路径（默认 dist\dshcam-v1.apk）
  -Adb  adb.exe 路径（默认 <AndroidSdk>\platform-tools\adb.exe）
  没检测到设备时打印中文排查清单并 exit 1。

.EXAMPLE
  pwsh -File .\scripts\install_apk.ps1 -Apk "D:\spider\tools\android-camera\dist\dshcam-v1.apk" -AndroidSdk "C:\Android\Sdk"

.NOTES
  本文件保存为 UTF-8（带 BOM），Windows PowerShell 5.1 与 PowerShell 7 都能正确显示中文。
#>
[CmdletBinding()]
param(
    [string]$Apk        = "D:\spider\tools\android-camera\dist\dshcam-v1.apk",
    [string]$Adb        = "",
    [string]$AndroidSdk = $env:ANDROID_HOME
)

$ErrorActionPreference = "Stop"

function Fail([string]$msg) {
    Write-Host ""
    Write-Host "[失败] $msg" -ForegroundColor Red
    exit 1
}

# ---------- 1. 检查 APK ----------
if (-not (Test-Path -LiteralPath $Apk)) {
    Fail "找不到 APK：$Apk`n先构建：pwsh -File scripts\build_apk.ps1 -AndroidSdk `"C:\Android\Sdk`" -JdkHome `"C:\...\jdk-17`""
}
$Apk = (Resolve-Path -LiteralPath $Apk).Path
Write-Host "[1/3] APK：$Apk"

# ---------- 2. 定位 adb ----------
if ([string]::IsNullOrWhiteSpace($Adb)) {
    if (-not [string]::IsNullOrWhiteSpace($AndroidSdk)) {
        $Adb = Join-Path $AndroidSdk "platform-tools\adb.exe"
    } else {
        $Adb = "adb"
    }
}

$adbExe = $Adb
if ($Adb -match '[\\/]') {
    if (-not (Test-Path -LiteralPath $Adb)) {
        $alt = Get-Command adb -ErrorAction SilentlyContinue
        if ($null -ne $alt) {
            $adbExe = $alt.Source
        } else {
            Fail "找不到 adb：$Adb`n请用 -Adb `"<SDK>\platform-tools\adb.exe`" 指定，或加 -AndroidSdk 指定 SDK 目录。"
        }
    }
} else {
    $alt = Get-Command $Adb -ErrorAction SilentlyContinue
    if ($null -eq $alt) {
        Fail "PATH 里找不到 adb 命令。请加 -Adb `"<SDK>\platform-tools\adb.exe`" 或 -AndroidSdk `"C:\Android\Sdk`"。"
    }
    $adbExe = $alt.Source
}
Write-Host "[2/3] adb：$adbExe"

# ---------- 3. 找设备 ----------
$deviceIds = @()
$adbOut = & $adbExe devices 2>&1 | Out-String
Write-Host "      adb devices 输出："
Write-Host ($adbOut.Trim()) -ForegroundColor DarkGray
foreach ($line in ($adbOut -split "`r?`n")) {
    if ($line -match '^\s*(\S+)\s+device\s*$') { $deviceIds += $Matches[1] }
}

if ($deviceIds.Count -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host " 没检测到手机（adb devices 列表为空）" -ForegroundColor Yellow
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host "请依次检查："
    Write-Host "  1. 用数据线把小米 14 Ultra 连到电脑；手机上弹出「允许 USB 调试吗？」要选「允许」"
    Write-Host "  2. 手机打开开发者选项：设置 → 关于手机 → 连点「OS 版本」7 次"
    Write-Host "  3. 设置 → 更多设置 → 开发者选项 → 打开「USB 调试」"
    Write-Host "  4. 小米/HyperOS 还必须打开「USB 安装」（开发者选项里；个别版本叫「USB 调试（安全设置）」）"
    Write-Host "  5. 数据线要能传数据（有的线只供电）；换一个 USB 口再试"
    Write-Host "  6. 如果上面列表里显示 unauthorized：在手机上点「允许 USB 调试」后重跑本脚本"
    Write-Host "  7. 小米还需要在「开发者选项」里把「USB 调试」相关开关全部打开，并关闭「MIUI 优化」的情况极少见"
    Write-Host ""
    exit 1
}

Write-Host "[3/3] 检测到 $($deviceIds.Count) 台设备：$($deviceIds -join ', ')"

$failed = $false
foreach ($id in $deviceIds) {
    Write-Host "     正在安装到 $id ..."
    & $adbExe -s $id install -r $Apk
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}

if ($failed) {
    Write-Host ""
    Write-Host "[失败] adb install 返回非 0。" -ForegroundColor Red
    Write-Host "常见原因与处理："
    Write-Host "  · 手机上弹出「是否安装此应用」→ 点「继续安装 / 仍要安装」"
    Write-Host "  · 弹出纯净模式 / 风险提示 → 点「仍要安装」（或先在 设置 → 应用设置 → 纯净模式 里临时关闭）"
    Write-Host "  · INSTALL_FAILED_USER_RESTRICTED → 打开开发者选项里的「USB 安装」"
    Write-Host "  · 已装过同包名但签名不同 → 先卸载旧版：`"$adbExe`" uninstall cn.yege.dshcam"
    exit 1
}

Write-Host ""
Write-Host "[成功] 已安装到手机。请在桌面找到「DSH 构图相机」打开；首次启动要允许「相机」权限。" -ForegroundColor Green
