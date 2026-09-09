param([Parameter(Mandatory=$true)][string]$Jobs)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $items = Get-Content -LiteralPath $Jobs -Raw | ConvertFrom-Json
    foreach ($job in $items) {
        $synth.SelectVoice($job.voice)
        $synth.Rate = [int]$job.rate
        $synth.Volume = 85
        $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
        $synth.SetOutputToWaveFile($job.path, $format)
        $synth.Speak([string]$job.text)
        $synth.SetOutputToNull()
        Write-Output $job.audio_id
    }
} finally { $synth.Dispose() }
