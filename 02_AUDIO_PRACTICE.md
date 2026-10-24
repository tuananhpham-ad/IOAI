# Audio — Học và thực hành theo dạng bài

> Trục bắt buộc: tín hiệu → processor/feature extractor → pretrained encoder/ASR/audio-LM → evaluation → tuning → inference. Dùng đúng model card/processor của checkpoint.

> Bổ sung có nguồn ngày **2026-10-24**: AUD-BS-01 → AUD-BS-07 gồm preprocessing, augmentation theo task, model cực nhỏ, ensemble/TTA, distillation và lab ablation. Thông số model lấy từ nguồn liên kết; recipe là đề xuất để thử, chưa phải kết quả benchmark trong repo.

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

## AUD-BS-01 — Preprocessing theo task (P0)

**Pipeline:** decode → kiểm tra channels/range → resample theo checkpoint → segment nếu cần → waveform augmentation khi train → processor/log-mel → feature augmentation khi train → batch/mask. Chia split theo speaker/recording/session trước khi tạo chunks hoặc augmented copies.

| Tình huống | Cách thử | Điều phải giữ đúng |
|---|---|---|
| Stereo / nhiều micro | So chọn kênh sạch với downmix; giữ spatial channels nếu task cần | Hai kênh ngược pha có thể triệt tiêu khi lấy mean; không luôn ép mono |
| Sample rate / codec | Decode một lần, resample thật bằng bộ lọc phù hợp; cache nếu được phép | Upsample 8 kHz lên 16 kHz không khôi phục dải tần bị mất; tránh resample lặp |
| Gain / loudness | So raw với normalization nhẹ, kiểm tra peak và clipping | Peak-normalize clip gần im lặng có thể khuếch đại noise; normalization không sửa được clipping |
| Silence / VAD cho ASR | So không VAD với VAD có padding đầu/cuối; thử padding 100–300 ms | Đây là range thực hành; nghe âm đầu/cuối bị cắt; lưu offset về recording gốc |
| Audio sự kiện/nhạc | Sliding windows hoặc event-aware crop | VAD phát hiện giọng nói có thể xóa tiếng chim, còi, nhạc; không dùng làm filter mặc định |
| Denoise / dereverb | Một nhánh ablation nếu noise thực sự gây lỗi | Có thể làm mất phụ âm, giọng nhỏ hoặc âm thanh mang nhãn; nghe và chấm sau biến đổi |
| Log-mel | Lấy `sr, n_fft, hop, n_mels, fmin/fmax, log scaling` từ checkpoint | Không thay số mel bins để “nhẹ hơn” mà không kiểm tra tương thích và fine-tune |
| Variable length | Bucket theo duration, pad và truyền lengths/masks đúng API | Pooling dùng valid frames sau encoder downsampling; không giả định input mask dùng trực tiếp được |
| Chunk ASR | Cắt tại khoảng nghỉ, overlap vừa đủ, align và bỏ phần lặp khi ghép | Không gán transcript toàn recording cho một crop; cần timestamps/alignment hoặc đoạn đã gán nhãn |
| Temporal event labels | Biến đổi onset/offset cùng waveform | Speed, crop, shift làm thay đổi thời gian; clip labels và frame labels không tương đương |

**Chuẩn hóa transcript:** lưu cả raw và normalized text; cố định Unicode, case, dấu câu, số, tiếng đệm theo scorer. Với tiếng Việt, cách tách theo khoảng trắng và cách word segmentation cho ra WER khác nhau; báo rõ tokenizer/scorer, giữ dấu và thêm CER. Không so hai WER dùng hai normalization khác nhau.

Ví dụ processor đặc thù: [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) dùng mono 16 kHz, 64 mel bins và embedding 1024 chiều; [HuBERT base](https://huggingface.co/facebook/hubert-base-ls960) nhận speech 16 kHz nhưng checkpoint base chưa có ASR tokenizer/head đã fine-tune. Không dùng cùng một tensor log-mel cho cả hai.

**Lab `LAB-AUD-07`:** baseline raw → + resample/processor đúng → + VAD riêng cho ASR → + denoise nếu có lý do. Mỗi run ghi quality, thời lượng trước/sau, tỷ lệ bị cắt và elapsed; nghe ít nhất 20 đoạn gồm speech nhỏ, silence, overlap và noise. Chỉ giữ bước làm tốt hơn metric mục tiêu.

## AUD-BS-02 — Augmentation chi tiết và điều kiện bảo toàn nhãn (P0/P1)

Các mức dưới đây là **candidate starting ranges**, không phải tối ưu đã đo. Bắt đầu với một hoặc hai phép nhẹ; tăng mức chỉ khi clean/noisy validation cho thấy có ích. Giữ validation chuẩn không augment ngẫu nhiên.

| Phép biến đổi | Khởi đầu thực hành | Task phù hợp / bẫy |
|---|---|---|
| Gain | ±3–6 dB, `p≈0.3` | ASR/classification; kiểm tra peak sau trộn, tránh clipping |
| Additive background noise | SNR `[10,20,30]` dB, `p≈0.3` | Noise thật cùng domain thường đáng thử; không lấy noise từ val/test recordings |
| Room impulse response (RIR) | Reverb nhẹ, `p≈0.2` | Speech xa micro; bù delay và giữ tail cần thiết; không cắt mất speech sau convolution |
| Speed perturb | `[0.9,1.0,1.1]` | Kiểu thay tốc độ phát làm đổi cả duration lẫn pitch; thận trọng tiếng Việt có thanh điệu, speaker/emotion/music |
| Time stretch giữ pitch | `[0.95,1.05]` trước | ASR/keyword khi lời còn rõ; thuật toán có thể tạo artifacts; cập nhật timestamps |
| Pitch shift giữ duration | ±1 semitone trước | Chỉ thử khi nhãn bất biến với pitch; thường rủi ro cho tone, speaker ID, emotion, nhạc |
| Time shift | ±50–100 ms, pad zero thay vì wrap mặc định | Keyword/event classification; cập nhật frame labels; wrap có thể cắt từ rồi nối hai đầu |
| Crop / pad | Theo cửa sổ model và vị trí event | Clip label chỉ giữ nếu crop còn event; ASR phải crop transcript tương ứng |
| Band-limit / codec simulation | Mô phỏng điện thoại hoặc codec thực tế | Telephony/noisy domain; không thêm artifact mạnh vào mọi mẫu clean |
| SpecAugment | 1–2 time masks ngắn, 1 frequency mask nhỏ, bắt đầu ≤5–10% mỗi trục | Trên feature phù hợp; không che hết utterance ngắn; time masking không phải xóa frame |
| Waveform / feature MixUp | `α=[0.2,0.4]` cho classification | Trộn soft targets với loss phù hợp; không nối hoặc trộn transcript ASR bằng công thức nhãn classification |
| Background/event mixing | Trộn với SNR và target tương ứng | Nếu thêm còi vào audio thì nhãn multi-label có thể phải thêm còi; không gọi mọi âm thanh là noise vô nhãn |
| Pseudo-label / self-training | Teacher trên unlabeled được phép; lọc theo confidence, duration và lỗi lặp | Ngưỡng confidence không dùng chung giữa model; cân bằng classes/accent; không đưa holdout vào train |

Nguồn phương pháp/triển khai: [SpecAugment paper](https://arxiv.org/abs/1904.08779) và [audiomentations: noise, gain, RIR, stretch, pitch, codec transforms](https://iver56.github.io/audiomentations/). SpecAugment gốc gồm cả time warping; không bắt buộc bật mọi thành phần khi fine-tune. Với HuBERT waveform input, dùng waveform transforms hoặc masking nội bộ đúng model thay vì đưa spectrogram vào waveform encoder.

**Noise theo SNR:** với signal `x`, noise `n` đã crop/pad cùng độ dài, chọn `a = RMS(x) / (RMS(n) * 10**(snr_db/20))`, rồi `mixed = x + a*n`. Xử lý RMS bằng 0; với clip nhiều silence, cân nhắc đo RMS vùng speech. Nếu cần giảm peak, scale toàn bộ mixture để giữ tỷ lệ; clip cứng sẽ làm méo tín hiệu.

**Ba recipe để ablate:**

- **ASR:** gain + noise nhẹ → RIR → speed/time stretch → SpecAugment nếu pipeline hỗ trợ. Không random crop speech rồi giữ nguyên toàn transcript.
- **Keyword/sound classification:** crop giữ event + shift → noise/gain → SpecAugment → MixUp riêng. Single-label CE nhận soft labels sau MixUp; multi-label dùng BCE với targets phù hợp.
- **Speaker/emotion:** split theo speaker/session đúng mục tiêu; bắt đầu noise/gain nhẹ; tránh pitch/stretch mạnh nếu chúng xóa đặc trưng cần nhận dạng.

**Lab `LAB-AUD-08`:** A0 không augment; A1 noise/gain; A2 thêm RIR; A3 một phép speed hoặc SpecAugment. Giữ optimizer steps và exposure tương đương; lưu `parent_id, transform, seed, snr, speed`. Báo clean val, noisy val cố định, per-accent/per-class và thời gian train.

## AUD-BS-03 — Model nhỏ gọn và ứng viên hiện đại theo task (P0)

Tra cứu ngày **2026-10-24**; M = triệu tham số. **Model ASR, VAD và audio tagging không thay thế nhau.** Các baseline cũ vẫn hữu ích khi cực thiếu compute; “SOTA” cần gắn benchmark và ngôn ngữ. Đếm tổng model triển khai, không chỉ trainable params hoặc decoder.

### Audio classification / embeddings / VAD

| Model / nguồn | Kích thước | Vai trò và lựa chọn |
|---|---:|---|
| MFCC/log-mel statistics + LogisticRegression/SVM | Phụ thuộc feature dimension/classes | Baseline CPU; fit scaler trên train; hữu ích để kiểm tra data/scorer trước deep learning |
| DS-CNN nhỏ tự train trên log-mel | Tự đặt budget, ví dụ <1M params | Keyword/command ít classes; đây là thiết kế baseline, không phải checkpoint SOTA có sẵn |
| [EfficientAT `mn04_as`](https://github.com/fschmid56/EfficientAT) | 0.983M | AudioSet tagging/embeddings; ưu tiên thử khi cần pretrained model dưới 1M |
| [EfficientAT `mn05_as`](https://github.com/fschmid56/EfficientAT) | 1.43M | Bước tăng nhẹ từ `mn04_as`; đổi head nếu bộ nhãn riêng |
| [EfficientAT `dymn04_as`](https://github.com/fschmid56/EfficientAT) | 1.97M | Dynamic CNN; đo backend latency, không suy từ params |
| [EfficientAT `mn10_as`](https://github.com/fschmid56/EfficientAT) | 4.88M | Mốc chất lượng cao hơn trong cùng family; frozen embeddings rồi fine-tune |
| [YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet) | CNN MobileNetV1; đếm lại artifact được dùng | 521 sound events, embeddings; baseline TensorFlow, không phải speech-to-text |
| [HuBERT base LS960](https://huggingface.co/facebook/hubert-base-ls960) | Khoảng 95M backbone | Speech representation cho intent/emotion và transfer; lớn hơn nhiều so EfficientAT; thêm head/task training |
| [Silero VAD](https://github.com/snakers4/silero-vad) | JIT artifact khoảng 2 MB theo repo | Phát hiện vùng speech, hỗ trợ 8/16 kHz; **MB là dung lượng, không phải M params**; không nhận diện speaker hay transcript |

**Số tác giả công bố:** EfficientAT `mn04_as` đạt AudioSet mAP 43.2, `mn10_as` đạt 47.1 trong bảng repo. Đây là mAP theo thang 0–100, không phải accuracy và không phải kết quả trên dataset riêng; bảng dùng checkpoint tốt nhất, có thể khác trung bình nhiều runs trong paper. Nguồn: [EfficientAT evaluation table](https://github.com/fschmid56/EfficientAT).

### ASR: ưu tiên từ vài chục triệu tham số

| Checkpoint / nguồn | Params ≈ | Ngôn ngữ / cách chọn |
|---|---:|---|
| [Moonshine Tiny](https://huggingface.co/moonshine-ai/moonshine-tiny) | 27M | English; ASR nhỏ cho edge, so với Whisper tiny.en trên cùng audio |
| [Moonshine Streaming Tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny) | 34M | English, kiến trúc streaming thế hệ 2026; benchmark đúng streaming runtime |
| [Whisper tiny / tiny.en](https://github.com/openai/whisper) | 39M | `tiny` đa ngôn ngữ có tiếng Việt; `tiny.en` chỉ English; chất lượng có thể hạn chế khi noisy/accent |
| [Moonshine Base](https://huggingface.co/moonshine-ai/moonshine-base) | 61M | English, tăng size trong family gốc; cần tải đúng checkpoint và processor base |
| [Gipformer 1.5 68M RNNT](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt) | 68M | Ứng viên 2026 chuyên tiếng Việt, Zipformer Transducer, có ONNX; kiểm chứng trên domain riêng |
| [Whisper base](https://github.com/openai/whisper) | 74M | Baseline đa ngôn ngữ/Việt gọn; `.en` là biến thể English |
| [Parakeet TDT-CTC 110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | Khoảng 114M theo card | English; tên 110M là làm tròn; hybrid TDT/CTC, dùng đúng decoder/backend |
| [Whisper small](https://github.com/openai/whisper) | 244M | Mốc chất lượng lớn hơn tiny/base khi còn budget |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 600M | 25 ngôn ngữ châu Âu, **không có tiếng Việt trong danh sách hỗ trợ**; chọn cho throughput nếu đúng ngôn ngữ |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | HF hiển thị tổng khoảng 0.9B | ASR mới, 30 ngôn ngữ gồm tiếng Việt + 22 phương ngữ Trung; không dùng “0.6B” trong tên làm tổng footprint |

**Hai điểm dễ hiểu nhầm:** card Moonshine Streaming ghi đường chạy Transformers chưa triển khai streaming hoàn toàn hiệu quả; gọi `generate()` trên cả file chưa chứng minh latency streaming. Qwen3-ASR là model chuyên ASR, không thay trực tiếp Qwen-Audio cho mọi bài audio QA; forced aligner đi kèm là model riêng và danh sách ngôn ngữ của aligner không giống ASR. Nguồn: [Moonshine Streaming card](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny), [Qwen3-ASR card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).

**Về SOTA:** tác giả Gipformer công bố WER 4.25 trên VIVOS cho bản 68M, sau normalization của họ. Đây là một kết quả theo dataset/protocol cụ thể, chưa được tái lập ở repo này; không kết luận vượt mọi model trên mọi giọng Việt. Với Qwen3-ASR, tuyên bố SOTA trong phần giới thiệu chủ yếu nói về bản **1.7B**, không tự gán cho bản nhỏ. Xem [Gipformer evaluation](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt) và [Qwen3-ASR evaluation](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).

**Chọn nhanh:** sound events → EfficientAT dưới 1–5M; speech intent/emotion → so frozen audio embeddings với HuBERT; English ASR → Moonshine Tiny/Whisper tiny.en; Vietnamese ASR → Whisper tiny/base và Gipformer 68M; chỉ nâng lên Qwen3-ASR khi chất lượng cần thêm bù được footprint. Audio QA có thể thử ASR nhỏ → NLP nhỏ, nhưng transcript sẽ mất âm sắc, nhạc và non-verbal cues.

## AUD-BS-04 — Ensemble/TTA đúng cho từng output (P1)

| Cách | Recipe | Chi phí / điều kiện |
|---|---|---|
| Classification soft voting | `p = Σ w_i*p_i`, weights không âm và tổng 1; bắt đầu 2 model | Căn clip IDs/classes; softmax single-label, sigmoid multi-label; tune thresholds sau blend |
| Handcrafted + neural | MFCC statistics classifier + EfficientAT; hoặc CNN + speech encoder cho task phù hợp | Cần errors bổ sung nhau; calibrate raw margins trước khi trộn với probabilities |
| Seed/fold ensemble | 2 seeds hoặc 3 group folds trước | Tổng train/infer cost tăng; mọi crop cùng recording ở cùng fold |
| Multi-crop TTA | Center + vài temporal windows hợp lệ; aggregate thành clip prediction | Average có thể làm loãng sự kiện hiếm; so mean/max/attention trên val; không vote crop không có event như đủ evidence |
| Waveform TTA nhẹ | Original + gain/shift nhỏ khi task bất biến | Không mặc định pitch, noise hoặc speed TTA giúp ASR; tốn thêm forward và có thể mất lời |
| ASR ROVER | Align chuỗi từ của nhiều recognizer rồi vote trên mạng từ | Không vote từ theo index vì insert/delete làm lệch; cùng normalization, language và recording |
| ASR n-best rescoring | ASR tạo n-best, LM/domain scorer xếp hạng lại | Tune acoustic/LM weight và length penalty trên val; không sửa tên/số chỉ vì câu khác “trôi chảy” hơn |
| Cascade | ASR nhỏ chạy trước, model mạnh xử lý subset khó | Confidence khác model không cùng scale; tune routing, báo tỷ lệ fallback và worst-case latency |
| EMA / SWA / soup | Deploy một checkpoint averaged tương thích | Cùng architecture/head/init khi soup; BatchNorm stats có thể phải cập nhật trên train; không trộn weights Whisper với Parakeet |

Nguồn ASR fusion: [NIST ROVER](https://www.nist.gov/publications/post-processing-system-yield-reduced-word-error-rates-recognizer-output-voting-error). Với token-level ensemble, vocab/blank IDs và frame timing phải tương thích; không cộng logits Whisper, CTC và RNNT chỉ vì tất cả sinh transcript. ROVER kết hợp text là hướng khác với average posterior.

**OOF và calibration:** fit weights/meta-model trên predictions của mẫu không được base model train; fit preprocessing trong từng fold. Nếu dùng OOF để train stacking hoặc chọn nhiều weights, cần holdout riêng/nested validation để báo chất lượng cuối. Không dùng điểm trên chính tập chọn ensemble làm bằng chứng generalization.

**Lab `LAB-AUD-09`:** classification: single model → multi-crop → blend 2 model. ASR: single recognizer → decode/chunk tốt nhất → ROVER hoặc cascade nếu có budget. Báo quality và tổng elapsed; ensemble chỉ được giữ khi đạt giới hạn inference. Với `Find the Order`, chấm thêm điểm sắp xếp cuối, không chỉ WER.

## AUD-BS-05 — Nén model và tối ưu runtime (P1)

- **Frozen embeddings:** precompute trên train/val riêng rồi fit linear head. Nếu chỉ cache một view/clip, không gọi đó là online augmentation thay đổi mỗi epoch; có thể cache nhiều views với provenance nếu đủ disk.
- **Classification distillation:** teacher/ensemble sinh soft labels trên train; student EfficientAT/DS-CNN học CE + temperature-scaled KL cho single-label. Multi-label dùng objective phù hợp sigmoid/Bernoulli. Teacher cost là chi phí chuẩn bị, không biến mất khỏi budget training.
- **ASR sequence distillation:** teacher tạo transcripts đã lọc → train student. Teacher và student có thể khác tokenizer; frame/logit distillation trực tiếp đòi hỏi alignment và vocabulary tương thích. Giữ phần gold labels làm mốc, tránh student học hallucination của teacher.
- **Quantization:** đo FP32/FP16/BF16 so INT8 theo backend; INT4 chỉ khi kiến trúc và kernel thực sự hỗ trợ. Calibration PTQ từ train/calibration split; đo lại WER, CER và lỗi tên riêng/số sau export.
- **Whisper runtime:** [faster-whisper/CTranslate2](https://github.com/SYSTRAN/faster-whisper) hỗ trợ các chế độ quantization/batching; đây là implementation, không phải checkpoint mới nhỏ hơn. So cùng model, beam, VAD, language, hardware; materialize `segments` generator khi đo elapsed để không chỉ đo lúc tạo iterator.
- **Streaming:** reuse state/cache đúng backend; phân biệt processing latency với chunk buffering/lookahead/endpoint delay. Giảm chunk giúp phản hồi sớm nhưng có thể giảm context và chất lượng.
- **Giảm input trước tăng model:** VAD đúng task, batch theo tổng audio seconds, giảm padding, giới hạn generation lặp. Không tùy tiện giảm sample rate/mel bins nếu checkpoint chưa hỗ trợ.

Nguyên lý student học teacher/ensemble: [Knowledge distillation](https://arxiv.org/abs/1503.02531). Model soup và giới hạn weight averaging: [Model soups](https://arxiv.org/abs/2203.05482). Đây là các phương án cần ablate trên audio, không bảo đảm giữ nguyên chất lượng.

**Nhỏ phải đo đủ:** total params; artifact MB gồm encoder/decoder/tokenizer/LM/aligner nếu sử dụng; peak RAM/VRAM; end-to-end latency. `P × bits / 8` chỉ ước tính bytes của weights lý tưởng; optimizer, activations, KV/state cache và quantization scales không nằm trong số đó. LoRA giảm trainable params, không tự giảm base model cần load.

## AUD-BS-06 — Benchmark và lab chọn model nhỏ nhất đạt yêu cầu (P0)

| Lab | So sánh cần chạy | Output bắt buộc |
|---|---|---|
| `LAB-AUD-10`: sound classification | MFCC+LR → EfficientAT `mn04_as` frozen/head → partial FT → `mn10_as` nếu cần | Macro-F1 hoặc mAP đúng task, per-class errors, artifact MB, latency/clip |
| `LAB-AUD-11`: English ASR | Moonshine Tiny 27M vs Whisper tiny.en 39M; thêm Streaming Tiny nếu bài cần streaming | Cùng audio/scorer; WER/CER, RTF, peak memory; TTFT/finalization cho streaming |
| `LAB-AUD-12`: Vietnamese ASR | Whisper tiny/base vs Gipformer 68M; Qwen3-ASR làm mốc lớn hơn nếu đủ budget | WER/CER theo accent/noise/duration, lỗi tên/số; cùng normalization |
| `LAB-AUD-13`: nén triển khai | Best single → quantized hoặc distilled student | Quality delta, export/reload correctness, artifact MB, total runtime |

**Quy tắc đo:**

1. `RTF = elapsed_seconds / audio_seconds`, nhỏ hơn tốt hơn; `RTFx` là nghịch đảo khi cùng điều kiện đo. Batch throughput không đồng nghĩa latency một request. Đo cả decode file, resample, VAD, model và postprocess nếu đây là thời gian scorer tính.
2. Corpus WER = tổng substitutions/deletions/insertions chia tổng reference words; không lấy mean WER từng clip rồi gọi là corpus WER. Với reference rỗng, quy định riêng và báo hallucination/speech false positives trên silence.
3. Audio classification dùng macro-F1 cho single-label imbalance; multi-label nên thêm mAP/per-class AP. VAD đo false alarm/miss, thêm ảnh hưởng lên WER và audio seconds giữ lại.
4. Dùng cùng split/scorer/hardware, warm-up, threads/batch, dtype/backend; đồng bộ GPU khi timing. Báo cold-start riêng nếu contest tính cả load model. Nếu chênh nhỏ, bootstrap theo recording/speaker hoặc thêm seeds thích hợp.
5. Ghi `run_id, dataset_hash, split, seed, model_revision, sr, duration_policy, augment, decode, metric, total_params, trainable_params, artifact_mb, peak_ram/vram, train_seconds, infer_seconds, RTF, notes`.

**Bảng ablation:** baseline → preprocess → augmentation → decoding/aggregation → ensemble → student/quantized. Không điền số đo chưa chạy; cho phép kết luận preprocessing/augment/ensemble không cải thiện.

## AUD-BS-07 — Thứ tự thử khi thời gian ít

| Mục tiêu | Thứ tự đề xuất |
|---|---|
| Cực nhẹ, keyword/sound events | MFCC+linear hoặc DS-CNN → EfficientAT 0.983M → augmentation phù hợp → INT8 |
| Speech emotion/intent | Frozen embeddings + head → partial unfreeze; so audio-native với transcript+classifier nếu task cho phép |
| ASR English | Moonshine Tiny / Whisper tiny.en → đúng segmentation/decode → tăng size nếu quality chưa đạt |
| ASR tiếng Việt | Whisper tiny/base và Gipformer 68M → phân tích accent/noise → model lớn hơn hoặc cascade có kiểm chứng |
| Streaming | Streaming checkpoint + runtime đúng → chunk/endpoint tuning → đo TTFT, finalization, WER |
| Contest vài phút | Đúng input/output/scorer → baseline → một thay đổi có ích → đóng gói offline; không bắt đầu ensemble nhiều model chưa đo |

**Exit chung:** đủ weights/processor/label map/decoder assets, reload trong process mới, chạy offline trên clip ngắn/dài/silence/noisy, output đúng schema. Với bài [Find the Order](example/1_Find_the_Order/README.md), giữ mapping `clip_id → transcript → pairwise score → order`; chọn phương án theo grader cuối cùng và tổng runtime.

## AUD-11 — Preprocessing audio nâng cao (P0)

| Bước | Khi nào | Cạm bẫy |
|---|---|---|
| Resample đúng SR model (Whisper/HuBERT/wav2vec2: 16 kHz; PANNs: 32 kHz; AST: 16 kHz) | Luôn | Lấy SR từ `processor.feature_extractor.sampling_rate`, không hard-code |
| Downmix stereo → mono (mean) | Đa số model | Task định vị/không gian cần giữ kênh |
| Loudness normalize (peak −1 dBFS hoặc LUFS −23 với `pyloudnorm`) | Dataset nhiều nguồn ghi âm | Không normalize theo clip nếu độ to là đặc trưng nhãn |
| DC offset removal, high-pass 60–80 Hz | Ghi âm điện thoại, tiếng ù | Không cho nhạc bass/âm thanh tần thấp |
| VAD trim (`silero-vad`) | ASR, speaker, clip nhiều silence | Cắt mất phụ âm đầu/cuối → thêm padding 100–200 ms |
| Chunking cố định (vd. 5–10 s, overlap 50%) | Clip dài cho classification | Nhãn clip-level gán cho chunk có thể sai (chunk toàn silence) |
| Pad/crop về độ dài cố định khi train, random crop khi train, center/multi-crop khi test | Classification | Pad zeros nhiều → model học "độ dài" |
| Feature: log-mel (n_mels 64/128, hop 10 ms), MFCC cho baseline cổ điển | CNN/AST tự build | Dùng processor của checkpoint nếu là model pretrained |
| Chuẩn hóa feature (mean/std trên train) | Model tự train | Không fit trên val/test |
| Split theo speaker/recording/session | Gần như mọi bài speech | Cùng speaker ở train và val → điểm ảo |

```python
import torch, torchaudio
wav, sr = torchaudio.load("x.wav")                 # [C, N]
wav = wav.mean(0, keepdim=True)                    # mono
wav = torchaudio.functional.resample(wav, sr, 16_000)
wav = wav - wav.mean()
wav = wav / (wav.abs().max() + 1e-8) * 0.89        # peak ≈ -1 dBFS
# VAD
vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
get_ts, _, _, _, collect = utils
speech = collect(get_ts(wav[0], vad, sampling_rate=16_000, speech_pad_ms=150), wav[0])
```

## AUD-12 — Augmentation code mẫu (P1)

Bổ sung cho bảng AUD-08: dùng `audiomentations` (waveform, CPU) hoặc `torch-audiomentations` (GPU, theo batch); SpecAugment trên log-mel bằng `torchaudio.transforms`.

```python
from audiomentations import Compose, AddGaussianSNR, TimeStretch, PitchShift, Shift, Gain, \
    AddBackgroundNoise, ApplyImpulseResponse, Mp3Compression, BandPassFilter
aug = Compose([
    Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
    AddGaussianSNR(min_snr_db=10, max_snr_db=40, p=0.4),
    # AddBackgroundNoise(sounds_path="noise/", min_snr_db=5, max_snr_db=25, p=0.4),  # MUSAN/ESC-50
    # ApplyImpulseResponse(ir_path="rir/", p=0.3),                                     # reverb thật
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),   # tránh với emotion/speaker nếu pitch là nhãn
    Shift(min_shift=-0.2, max_shift=0.2, p=0.3),
    Mp3Compression(p=0.2), BandPassFilter(p=0.2),           # mô phỏng điện thoại/codec
])
y = aug(samples=wave_np, sample_rate=16_000)

import torchaudio.transforms as T
spec_aug = torch.nn.Sequential(T.FrequencyMasking(freq_mask_param=24), T.TimeMasking(time_mask_param=80))
```

Thêm: **audio Mixup** (trộn 2 waveform + trộn nhãn; với multi-label sound event thì lấy `max`/`union` nhãn), **speed perturb 0.9/1.0/1.1** cho ASR (Kaldi recipe), **pseudo-label** audio unlabeled bằng Whisper large / ensemble.

## AUD-13 — Model nhỏ gọn hiện tại (2025–2026) cho audio

> Kiểm tra model card, license, ngôn ngữ hỗ trợ, SR yêu cầu. Số tham số xấp xỉ.

### ASR

| Model | Params | Ghi chú |
|---|---|---|
| `UsefulSensors/moonshine-tiny` / `-base` | 27M / 61M | Tiếng Anh, nhanh hơn Whisper tiny, latency tỉ lệ độ dài audio |
| `openai/whisper-tiny` / `base` / `small` | 39M / 74M / 244M | Đa ngữ (có tiếng Việt); fine-tune dễ |
| `distil-whisper/distil-small.en` / `distil-large-v3.5` | 166M / 756M | Distill, nhanh ~5x, tiếng Anh |
| `openai/whisper-large-v3-turbo` | 809M | Decoder 4 layer, gần large-v3 nhưng nhanh hơn nhiều |
| `nvidia/parakeet-tdt-0.6b-v2` (en) / `-v3` (25 ngôn ngữ châu Âu) | 600M | Top Open ASR Leaderboard cỡ nhỏ, cực nhanh (NeMo); **không có tiếng Việt** |
| `nvidia/canary-180m-flash` | 182M | En/de/es/fr, ASR + dịch |
| `vinai/PhoWhisper-tiny/base/small` | 39M–244M | Whisper fine-tune tiếng Việt (nhiều giọng vùng miền) |
| `nguyenvulebinh/wav2vec2-base-vietnamese-250h` | 95M | CTC tiếng Việt, nhẹ |
| `mistralai/Voxtral-Mini-3B-2507` | 3B | ASR + audio QA (xem AUD-06) |

### Speech/audio encoder & classification

| Model | Params | Ghi chú |
|---|---|---|
| `fschmid56/EfficientAT` (mn04/mn10) | 1–5M | CNN mobile distill từ AST, AudioSet mAP cao so với kích thước |
| PANNs `CNN10` / `CNN14` (`panns-inference`) | 5M / 80M | Baseline sound event, 32 kHz |
| `MIT/ast-finetuned-audioset-10-10-0.4593` | 87M | Audio Spectrogram Transformer; fine-tune tốt |
| BEATs iter3 / `ATST`, `EAT-base` | ~90M | SSL audio, mạnh cho AudioSet/ESC-50 |
| `microsoft/wavlm-base-plus` | 95M | Speech encoder tốt nhất cỡ base (SUPERB): speaker, emotion, intent |
| `facebook/hubert-base-ls960` / `wav2vec2-base` | 95M | Xem AUD-02 |
| `emotion2vec/emotion2vec_plus_base` | ~90M | Speech emotion recognition, đa ngữ |
| `speechbrain/spkrec-ecapa-voxceleb` | ~20M | Speaker embedding/verification |
| `pyannote/speaker-diarization-3.1` | nhỏ | Diarization (ai nói khi nào) — hữu ích cho bài hội thoại |
| `laion/clap-htsat-unfused` / `microsoft/msclap` | ~150M | Zero-shot audio classification bằng text prompt |
| `snakers4/silero-vad` | ~2MB | VAD CPU |

### Audio LLM nhỏ

| Model | Params | Ghi chú |
|---|---|---|
| `google/gemma-3n-E2B-it` | ~5B (E2B hiệu dụng 2B) | Audio + image + text input, chạy được GPU nhỏ |
| `Qwen/Qwen2.5-Omni-3B` | 3B | Audio/video/text → text (+speech) |
| `microsoft/Phi-4-multimodal-instruct` | 5.6B | ASR/AST tốt, audio QA |
| `Qwen/Qwen2-Audio-7B-Instruct` | 7B | Xem AUD-05; cần 4-bit cho GPU 16 GB |

**Chiến lược chọn:** ASR: Whisper-small/PhoWhisper (vi) hoặc Parakeet/Moonshine (en) → turbo khi cần chính xác. Classification: embedding WavLM/BEATs/CLAP frozen + LR (phút) → fine-tune AST/EfficientAT → ensemble.

## AUD-14 — Brute force tăng điểm: ensemble + TTA tất tay [CONTEST]

> Giống NLP-14/CV-07: K-fold (split theo speaker), lưu OOF + test probabilities, hill-climbing trên OOF (code ở NLP-14), không nhìn test, đo tổng thời gian inference.

### Classification (sound event, emotion, speaker, intent)

1. **Đa dạng input**: waveform model (WavLM/HuBERT) + spectrogram model (AST/EfficientAT/PANNs) + CLAP/BEATs embedding + LR. Hai họ này sai khác nhau → ensemble lợi nhiều.
2. **Đa dạng feature**: n_mels 64 vs 128, hop 10 vs 20 ms, độ dài crop 5 s vs 10 s → mỗi biến thể là 1 model.
3. **Layer-weighted pooling**: học trọng số trên tất cả hidden layers của WavLM/HuBERT (SUPERB-style) thay vì chỉ layer cuối.
4. **Multi-seed + EMA + Mixup + SpecAugment**.
5. **Pseudo-label** + **distill** ensemble → 1 model nếu bị giới hạn latency.

| TTA | Cách làm | Gộp |
|---|---|---|
| Multi-crop / sliding window | Cắt clip test thành nhiều đoạn overlap 50% | mean (nhãn toàn clip) hoặc max (sự kiện ngắn có thể chỉ ở 1 đoạn) |
| Time shift | Dịch ±0.1–0.5 s | mean |
| Gain | ±3 dB | mean |
| Speed 0.95/1.05 | Chỉ khi train có speed perturb | mean |
| Thêm noise nhẹ | Chỉ khi test noisy giống train augment | mean (trọng số thấp) |

### ASR

| Cách | Ghi chú |
|---|---|
| Beam search 5 + length penalty, temperature fallback | Tune trên val WER |
| **ROVER** (`SCTK`) / voting trên word alignment giữa nhiều model (Whisper + Parakeet + PhoWhisper) | Ensemble ASR chuẩn; hoặc chọn hypothesis có nhiều đồng thuận nhất (MBR với WER) |
| **LLM rescoring / post-correction** N-best | Dùng LLM nhỏ sửa chính tả, tên riêng, dấu tiếng Việt; cẩn thận hallucination |
| Initial prompt / hotwords | `initial_prompt` của Whisper với tên riêng, thuật ngữ trong đề |
| Chunk với overlap + merge theo timestamp | Audio dài; tránh lặp từ ở biên |
| Text normalization đúng scorer | Thường thêm nhiều điểm WER hơn cả đổi model |

```python
# Multi-crop TTA cho audio classification
def predict_tta(model, wav, sr=16_000, win=10, hop=5):
    W, H = win * sr, hop * sr
    if wav.shape[-1] <= W:
        crops = [torch.nn.functional.pad(wav, (0, W - wav.shape[-1]))]
    else:
        crops = [wav[..., s:s + W] for s in range(0, wav.shape[-1] - W + 1, H)]
    with torch.no_grad():
        p = torch.stack([model(c.unsqueeze(0)).softmax(-1)[0] for c in crops])
    return p.mean(0)          # .max(0).values cho sound event ngắn
```

```python
# Ensemble ASR đơn giản: chọn hypothesis có tổng WER nhỏ nhất so với các hypothesis còn lại (MBR)
import jiwer
def mbr_pick(hyps):
    return min(hyps, key=lambda h: sum(jiwer.wer(o, h) for o in hyps if o is not h))
```

**Cảnh báo:** pitch shift/speed TTA hại với emotion/speaker; ROVER cần các model cùng normalization; ensemble ASR làm RTF tăng tuyến tính → kiểm tra giới hạn thời gian.
