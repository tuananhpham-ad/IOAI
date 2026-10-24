# Audio — Học và thực hành theo dạng bài

> Trục bắt buộc: tín hiệu → processor/feature extractor → pretrained encoder/ASR/audio-LM → evaluation → tuning → inference. Dùng đúng model card/processor của checkpoint.

## AUD-01 — Hiểu dữ liệu âm thanh (P0)

### Thuộc tính cần xác minh trước mọi lần train

- `sample_rate` (Hz): số mẫu/giây; **phải resample thật**, không đổi metadata từ 44.1 kHz thành 16 kHz.
- `num_channels`: mono/stereo; downmix cần thử nếu mất thông tin không gian.
- `num_samples`, `duration_sec = num_samples / sample_rate`; audio clip quá dài → OOM/encoder limit.
- `peak amplitude`, RMS, clip (nhiều mẫu sát ±1), khoảng silent, SNR ước lượng khi có noise reference.
- Container/codec, lỗi decode, dữ liệu duplicate, mismatch transcription/audio.

### Bài cơ bản

1. Load 20 file bằng `soundfile`/`librosa`/torchaudio (theo stack/version), chuẩn hóa float32 khi thích hợp.
2. Vẽ waveform, spectrogram/log-mel, histogram duration và class imbalance.
3. Resample 8k/44.1k → 16k, kiểm tra **duration được giữ**.
4. Pad/trim theo thời lượng, tạo mask hoặc `input_lengths` theo processor/collator; batch variable-length.
5. Nghe 10 mẫu và kiểm tra label sai, silent/noisy/clipped.

### Công thức (đừng nhầm)

- Seconds = `N / sr`; sau resample từ `sr_in` sang `sr_out`, kích thước kỳ vọng xấp xỉ `N * sr_out / sr_in`.
- Clip dài `L` giây ở 16 kHz có khoảng `L*16000` mẫu, **không phải** L token.
- Log-mel features có shape thường `[mel_bins, frames]`, tùy processor; không cứng hóa kích thước.

## AUD-02 — HuBERT encoder (P0)

### Kiểu bài 1: Frozen representation + classifier

`audio → AutoFeatureExtractor/AutoProcessor → HubertModel → hidden states [B,T,H] → masked temporal pooling → classifier`.

- HuBERT là pre-trained speech representation model; phù hợp speech tasks, không mặc định là tốt nhất cho nhạc/môi trường tự nhiên.
- Masked pooling phải dùng **độ dài valid sau downsampling** (hoặc mask được chuyển theo model). Không lấy mean trên padded frames.
- Nếu dataset nhỏ, bắt đầu frozen encoder + logistic regression / MLP head.
- Metrics: classification macro-F1, per-class accuracy, confusion; report latency/clip.

### Kiểu bài 2: Fine-tuning

- Giữ speech encoder frozen trong warm-up rồi unfreeze top layers; hoặc full FT nếu compute cho phép.
- Có thể dùng `AutoModelForAudioClassification` khi checkpoint/class phù hợp; nếu không, tự head theo API hiện hành.
- Tuning: LR encoder thấp hơn head (discriminative LR), layer unfreeze, pooling, duration/chunking, augmentation.
- **Lab `LAB-AUD-02`:** HuBERT frozen vs partial FT trên speech intent/emotion classification; tránh train/test chung speaker nếu bài yêu cầu generalize sang speaker mới.

## AUD-03 — Whisper ASR inference (P0)

- Pipeline: waveform → processor log-mel → encoder-decoder → transcript; xác nhận language, `task=transcribe` vs `translate`.
- So sánh output chuẩn hóa và nguyên văn; normalize phải phù hợp ngôn ngữ và yêu cầu chấm điểm.
- Metrics: WER (word-level), CER (char-level); nghe đối chiếu false insertions, omitted words, tên riêng, numbers, code-switching.
- Với audio dài: VAD/segmentation/chunk strategy, overlap, xử lý nối câu và duplicate; **không** mặc định chunking không làm sai ngữ cảnh.
- Decode: beam size, temperature fallback (nếu API hỗ trợ), language forcing, timestamps, condition_on_previous_text tùy implementation; ghi chính xác wrapper.
- **Lab `LAB-AUD-03`:** baseline `openai/whisper-*` hoặc `faster-whisper`, evaluate tập ngắn có transcript; benchmark WER, CER, real-time factor (RTF), peak VRAM.

## AUD-04 — Whisper fine-tuning (P0)

### End-to-end workflow

1. Data schema `(audio, text, speaker_id?, recording_id?)`; kiểm tra license và empty/noisy labels.
2. Audio decoding và *resampling đúng* theo model processor.
3. Text normalization chỉ theo rule phù hợp bài: punctuation/case/numbers/diacritics được giữ hay bỏ?
4. `WhisperProcessor` xử lý feature và labels; data collator pad audio features và decoder labels riêng; mask pad labels `-100`; xử lý BOS khi cần.
5. `WhisperForConditionalGeneration`/API tương thích; `Seq2SeqTrainer` hoặc training loop; evaluate bằng `predict_with_generate`.
6. Chọn best checkpoint theo **validation WER/CER**, không nhìn test; so sánh với zero-shot baseline.

### Biến cần tune

| Nhóm | Thử nghiệm | Ghi chú |
|---|---|---|
| Model | tiny/base/small/medium tùy GPU | RTF, VRAM, chất lượng khác nhau; không mặc định tăng size luôn lời |
| Training | LR `1e-5`, `3e-5` (khởi đầu), steps, warmup, scheduler | phụ thuộc model/dataset; theo dõi WER chứ không chỉ loss |
| Duration | bỏ silence quá dài, group_by_length, batch by duration | tránh cắt mất chữ |
| Transcript | normalization policy, label fixes | WER phụ thuộc normalization |
| Data aug | low noise, speed perturb, mild reverberation | validation cần giống hidden distribution |
| Decode | beam width 1/3/5, language, temperature | so sánh quality-latency; không tune test |
| Resource | bf16/fp16, accumulation, grad checkpoint | phần cứng phụ thuộc |

**Lab `LAB-AUD-04`:** fine-tune Whisper trên tập nhỏ hợp pháp (HF Audio Course hướng dẫn Common Voice), thực hiện ablation *zero-shot vs FT*, báo cáo WER, RTF, sample output. Hướng dẫn HF có ví dụ phù hợp GPU 16 GB, nhưng thời gian/VRAM thực tế phụ thuộc cấu hình.

## AUD-05 — Qwen-Audio (P1)

- **Qwen-Audio** (repo gốc) là multimodal **audio-input → text-output**, hỗ trợ đa tác vụ audio understanding, audio question answering, sound/event description và speech.
- Phân biệt Qwen-Audio bản gốc với **Qwen2-Audio** hay Qwen2.5-Omni: model ID, processor, chat template và thư viện yêu cầu không giống nhau.
- Ưu tiên inference đúng prompt/template từ repo hoặc model card; không sao chép code từ model đời khác.
- Task practice: (1) nhận diện sự kiện âm thanh, (2) Q&A dựa trên audio, (3) caption, (4) so sánh transcript→text model với trực tiếp audio model.
- Lưu ý memory, context/audio length, output parsing và hỗ trợ CPU/GPU; 7B và audio LLM nói chung có thể quá nặng cho GPU 16 GB không quantize.
- **Lab `LAB-AUD-05`:** run 10–30 clip ngắn, đánh giá task correctness, hallucination và latency; thêm rule không được suy diễn thông tin không có trong audio.

## AUD-06 — Voxtral (P1)

- Voxtral là dòng model audio của Mistral; **không giả định tất cả phiên bản có cùng khả năng** (ASR, audio QA, streaming, speaker label, timestamps, TTS là các khả năng phụ thuộc model/version).
- Nếu dùng API: đọc docs Mistral Audio/Voxtral, xác định endpoint, payload, file format, model ID, giá/quota; không in hoặc ghi API key.
- Nếu dùng checkpoint mở: xác minh weights, processor/code, giấy phép, GPU trước khi bắt đầu.
- Benchmark ít nhất: transcripts/answers, word errors hoặc task correctness, elapsed/RTF, lỗi theo noise/clip length.
- **Lab `LAB-AUD-06`:** chạy cùng một bộ audio trên Whisper và phiên bản Voxtral phù hợp; ghi rõ điều kiện so sánh (model, API/local, normalization, latency hardware/network).

## AUD-07 — Audio → NLP pipeline & contest tasks

### Speech-to-text + downstream reasoning

- Baseline: Whisper → transcript → sentence embeddings/LLM/classifier.
- Oracle: gold transcript → downstream model. Chênh lệch oracle–ASR ước lượng lỗi truyền qua pipeline.
- Audio-native: HuBERT/Audio-LLM → label/answer. So sánh khi tiếng ồn, speaker turn, non-verbal cues.

### IOAI 2026 `Find the Order`

Bài toán chính thức: hội thoại tiếng Anh chia thành file WAV của từng lượt nói và bị xáo trộn; cần tái tạo thứ tự lượt. Repo cung cấp original baseline/grader và educational notebook có data từ Hugging Face.

**Hướng giải thử nghiệm:**

1. Kiểm tra WAV và nhãn/metadata có sẵn (theo luật bài).
2. Transcribe từng segment bằng ASR hoặc model được phép dùng.
3. Tính pairwise transition score giữa lượt i→j: semantic coherence, dialogue acts, question→answer compatibility, punctuation, speaker turn constraints.
4. Tìm chuỗi có tổng score cao (greedy/beam search/DP/optimization tùy số lượt), so với heuristic baseline.
5. Report exact sequence accuracy hoặc điểm grader chính thức, inference seconds và memory; không tự giả định cách chấm.

Điểm đặc biệt: **chỉ fine-tune Whisper không trực tiếp giải toàn bài**; phải mô hình hóa chuyển tiếp hội thoại và sắp xếp thứ tự.

## AUD-08 — Audio augmentations có kiểm soát [EXTENSION]

| Technique | Tác dụng | Tránh khi |
|---|---|---|
| Gain/loudness jitter | Robustness khác mức âm lượng | Artifact/clipping |
| Additive noise (SNR-controlled) | Chống noise background | Noise che nhãn/signal trọng yếu |
| Speed perturb (0.9–1.1) | Accent/tempo variation | Task phụ thuộc pitch/nhịp chính xác |
| Time shift/crop | Invariance vị trí | Mất key event hoặc speech |
| Reverberation | Robustness môi trường | Reverb không hợp distribution |
| Time/freq masking (SpecAugment) | Regularize features | Không đúng representation hoặc quá mạnh |
| Silence trim / VAD | Giảm tính toán | Cắt mất lời nhỏ/âm cuối |
| Mixup (label-compatible) | Regularize classifier | ASR cần transcript chính xác; labels không tuyến tính |
| Pseudo-label ASR | Mở rộng train data | Nhiễm lỗi hệ thống từ teacher |

*Chỉ bật augmentation trên training*, và quyết định qua validation thực tế. Không dùng augment làm rò rỉ near-duplicate từ test.

## AUD-09 — Audio error analysis template

Mỗi run lấy ít nhất 15 lỗi và phân nhóm: bad audio/codec; low volume; noise; accent; code-switching; tên riêng/number; VAD cắt chữ; speaker overlap; long duration; hallucinated transcription; label mismatch; prompt/template error; decode/rate mismatch; OOM/truncation. Ghi một sửa đổi có thể thử và đo lại.

## AUD-10 — Mẫu code ngắn kiểm tra resampling

```python
import soundfile as sf
import librosa

wave, sr = sf.read('sample.wav', dtype='float32', always_2d=True)
mono = wave.mean(axis=1)
print('before', sr, len(mono) / sr)

TARGET_SR = 16_000  # chỉ dùng nếu model/processor cần 16 kHz
resampled = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR)
print('after ', TARGET_SR, len(resampled) / TARGET_SR)

assert abs(len(mono)/sr - len(resampled)/TARGET_SR) < 0.02
```

**Không được** viết `sr = 16000` nếu waveform vẫn ở sample rate cũ.
