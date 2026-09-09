param([string]$AudioPath,[string]$OutputPath)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$recognizer=New-Object System.Speech.Recognition.SpeechRecognitionEngine('MS-1033-80-DESK')
try {
    $recognizer.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
    $recognizer.SetInputToWaveFile($AudioPath)
    $parts=New-Object System.Collections.Generic.List[string]
    while ($true) {
        try { $result=$recognizer.Recognize() }
        catch {
            if ($parts.Count -gt 0 -and $_.Exception.Message -like '*No audio input*') { break }
            throw
        }
        if ($null -eq $result) { break }
        $parts.Add($result.Text)
    }
    [System.IO.File]::WriteAllText($OutputPath,($parts -join ' '),[System.Text.UTF8Encoding]::new($false))
} finally { $recognizer.Dispose() }
