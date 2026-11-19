param(
    [Parameter(Mandatory=$true)][string]$Packages,
    [Parameter(Mandatory=$true)][string]$Video,
    [Parameter(Mandatory=$true)][string]$Audio,
    [Parameter(Mandatory=$true)][ValidateSet('cpu','cuda','vulkan')][string]$Backend,
    [Parameter(Mandatory=$true)][string]$Device,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Runner
)

$ErrorActionPreference = 'Stop'
$runnerPath = $null
if ($Runner) { $runnerPath = (Resolve-Path -LiteralPath $Runner).Path }
$settings = @{
    FT_SMOKE_PACKAGES = (Resolve-Path -LiteralPath $Packages).Path
    FT_SMOKE_VIDEO = (Resolve-Path -LiteralPath $Video).Path
    FT_SMOKE_AUDIO = (Resolve-Path -LiteralPath $Audio).Path
    FT_SMOKE_BACKEND = $Backend
    FT_SMOKE_DEVICE = $Device
}
if (Test-Path -LiteralPath $Output) { throw "Choose a new output directory: $Output" }
$settings.FT_SMOKE_ROOT = (New-Item -ItemType Directory -Path $Output).FullName
$previous = @{}
foreach ($entry in $settings.GetEnumerator()) {
    $previous[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, 'Process')
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
$code = 1
try {
    [PSCustomObject]@{
        os = (Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber)
        cpu = @(Get-CimInstance Win32_Processor | Select-Object Name)
        display_adapters = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $settings.FT_SMOKE_ROOT 'system.json')
    $ErrorActionPreference = 'Continue'
    $log = Join-Path $settings.FT_SMOKE_ROOT 'smoke.log'
    if ($runnerPath) {
        & $runnerPath train_cancel_reopen_resume_export --ignored --nocapture 2>&1 |
            ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    } else {
        Push-Location (Join-Path $PSScriptRoot '../..')
        try {
            cargo test --offline --locked -p feather-talk-gui --test hardware_acceptance train_cancel_reopen_resume_export -- --ignored --nocapture 2>&1 |
                ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
            $code = $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }
} finally {
    foreach ($entry in $previous.GetEnumerator()) { [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process') }
}
exit $code
