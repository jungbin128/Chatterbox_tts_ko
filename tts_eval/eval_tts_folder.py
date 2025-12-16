import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from jiwer import wer
from wvmos import get_wvmos
import whisper


def get_audio_files(audio_dir: Path, exts=None) -> List[Path]:
    if exts is None:
        exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    files = [p for p in audio_dir.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def evaluate_folder(
    audio_dir: Path,
    whisper_model_name: str = "large-v2",
    language: Optional[str] = None,
    use_gpu_for_mos: bool = True,
):
    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    print("Loading WV-MOS model...")
    mos_model = get_wvmos(cuda=use_gpu_for_mos)

    print(f"Loading Whisper model: {whisper_model_name} ...")
    asr_model = whisper.load_model(whisper_model_name)

    records = []

    for idx, audio_path in enumerate(audio_files, start=1):
        print(f"[{idx}/{len(audio_files)}] Processing {audio_path.name} ...")

        text_path = audio_path.with_suffix(".txt")

        mos_score = float(mos_model.calculate_one(str(audio_path)))

        if language:
            result = asr_model.transcribe(str(audio_path), language=language)
        else:
            result = asr_model.transcribe(str(audio_path))
        asr_text = result.get("text", "").strip()

        if text_path.exists():
            ref_text = text_path.read_text(encoding="utf-8").strip()
            if len(ref_text) == 0:
                sample_wer = None
                print("  -> WARNING: reference text empty. WER skipped.")
            else:
                sample_wer = wer(ref_text, asr_text)
        else:
            ref_text = None
            sample_wer = None
            print(f"  -> WARNING: reference text file not found: {text_path.name}. Skipping WER.")

        records.append(
            {
                "file": audio_path.name,
                "audio_path": str(audio_path),
                "ref_text_path": str(text_path) if text_path.exists() else None,
                "reference_text": ref_text,
                "asr_text": asr_text,
                "mos": mos_score,
                "wer": sample_wer,
            }
        )

    df = pd.DataFrame(records)

    mean_mos = df["mos"].mean()
    mean_wer = df["wer"].dropna().mean() if df["wer"].notna().any() else None

    print("\n=== Per-file results ===")
    print(df[["file", "mos", "wer"]].to_string(index=False))

    print("\n=== Summary ===")
    print(f"Average MOS: {mean_mos:.4f}")
    if mean_wer is not None:
        print(f"Average WER: {mean_wer:.4f}")
    else:
        print("Average WER: N/A (no valid reference texts)")

    out_csv = audio_dir / "tts_eval_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved detailed results to: {out_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTS audio folder with MOS (WV-MOS) + Whisper WER.")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Path to folder containing audio files and their matching .txt files.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v2",
        help='Whisper model name ("tiny", "base", "small", "medium", "large-v2", "large-v3", etc.)',
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help='Language hint for Whisper (e.g. "Korean", "English"). If omitted, auto-detect.',
    )
    parser.add_argument(
        "--cpu_mos",
        action="store_true",
        help="Force MOS to run on CPU.",
    )

    args = parser.parse_args()
    audio_dir = Path(args.audio_dir)

    if not audio_dir.is_dir():
        raise ValueError(f"{audio_dir} is not a directory")

    evaluate_folder(
        audio_dir=audio_dir,
        whisper_model_name=args.whisper_model,
        language=args.language,
        use_gpu_for_mos=not args.cpu_mos,
    )


if __name__ == "__main__":
    main()
