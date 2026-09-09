"""Optional Whisper backend. Reference transcripts never enter recognition."""
class WindowsSTT:
    def transcribe(self,path):
        import subprocess
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix='stage3_stt_') as tmp:
            output=Path(tmp)/'transcript.txt'
            subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                str(Path(__file__).with_name('recognize.ps1')),'-AudioPath',str(path),'-OutputPath',str(output)],check=True,timeout=300)
            return output.read_text(encoding='utf-8')

class WhisperSTT:
    def __init__(self,name='tiny.en',device='cpu',download_root=None):
        from faster_whisper import WhisperModel
        self.name=name
        self.model=WhisperModel(name,device=device,compute_type='int8' if device=='cpu' else 'float16',
                                download_root=download_root,cpu_threads=4)

    def transcribe(self,path):
        segments,info=self.model.transcribe(str(path),language='en',beam_size=5,
            vad_filter=False,condition_on_previous_text=False)
        return ' '.join(segment.text.strip() for segment in segments)
