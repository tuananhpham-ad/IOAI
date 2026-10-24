# TUNING PLAYBOOK — NLP + Audio (đọc trước khi thay hyperparameters)

> Mục tiêu: có hệ thống tìm phương án tăng **validation quality** dưới giới hạn **runtime / VRAM / inference**, không chỉ tăng train accuracy. Các khoảng giá trị ở đây là *candidate starting ranges*, phải thích ứng với dataset/model/hardware.

## 0. Protocol bắt buộc

1. Đặt `PRIMARY_METRIC`: NLP classification = macro-F1 nếu lệch nhãn; ASR = WER/CER; retrieval = MRR/Recall@k; MT = chrF/BLEU; audio classification = macro-F1.
2. Cố định dữ liệu train/val/test, normalization/scorer, seed. Nếu data theo speaker/user/book/source, dùng group split phù hợp.
3. Chạy baseline tối thiểu: naive/lexical/frozen/pre-trained inference; ghi `metric, seconds, GPU peak memory`.
4. Chạy **một thay đổi** hoặc thí nghiệm có thiết kế, giữ tất cả thứ khác, ghi config + seed + result.
5. Chọn config qua validation; test chỉ dùng báo cáo cuối. Nếu public leaderboard chỉ cho feedback giới hạn, tránh brute-force theo leaderboard.
6. Báo cáo cả mean±std khi khác biệt nhỏ so với variance do seed. Đừng nói "improved" nếu chưa đo.

## 1. Tuning dữ liệu — thường mang lại nhiều lợi ích hơn đổi optimizer

| Yếu tố | NLP | Audio | Kiểm chứng |
|---|---|---|---|
| Label quality | Null, ambiguous, inconsistent label; duplicate | Transcript lệch, annotator errors, incorrect clip label | Manual audit tối thiểu 20 examples/class nếu đủ |
| Split strategy | Stratified, time/group/domain held-out | Speaker/session/recording held-out | Leak check sau split, near-duplicate hash |
| Imbalance | Class weights, balanced batches, focal loss (thử có căn cứ) | Macro-F1, per-class recall, oversampling | Không resample validation/test |
| Length | Long-context, smart chunk, head+tail | Durations, silence, VAD, chunk/overlap | Truncation rate + dropped content |
| Normalize | Whitespace/Unicode/casing phù hợp model | Text WER normalization + audio resample | Giữ nguyên proper nouns/diacritics nếu quan trọng |
| Augmentation | Paraphrase có bảo toàn nhãn; token masking thận trọng | Noise, speed, reverb, gain, time/freq mask | Chỉ train, so val clean/noisy |
| Domain match | Model phù hợp tiếng Việt/domain | Accent, language, mic/environment | Held-out domain/speaker scores |
| Sampling | Hard negatives có kiểm soát; balanced sampling | Balance durations/noise/speakers | Không dùng val/test labels để chọn examples |

**Cảnh báo:** với văn bản, lowercasing có thể phá phân biệt ký hiệu/tên riêng; với audio, nén, tăng tốc hoặc VAD quá mạnh có thể thay nội dung nhãn. Không tẩy metadata bằng cách lặng lẽ thay sample rate.

## 2. NLP classification — thứ tự tune ưu tiên

1. Check metrics/split/label mapping; TF-IDF (word/char n-grams) baseline.
2. Pretrained model và ngôn ngữ: DistilBERT/BERT/phù hợp tiếng Việt/multilingual.
3. Max length dựa trên p95 token length; batch + accumulation đáp ứng GPU.
4. Learning rate / epochs / warmup / early stopping.
5. Class weights/thresholds đúng chế độ binary/multi-label; focal loss nếu cần.
6. Dropout, weight decay, layer-wise learning rate decay, freeze/unfreeze.
7. Ensembling folds hoặc checkpoints chỉ khi budget cho phép.

**Search space nhỏ (điểm bắt đầu):**

| Hyperparameter | Candidate | Ghi chú |
|---|---|---|
| `learning_rate` full fine-tune encoder | `[1e-5, 2e-5, 3e-5, 5e-5]` | Đo divergence; không dùng vô điều kiện cho head-only |
| `num_train_epochs` | `[2, 3, 5]` | Đo val mỗi epoch; early stop |
| `weight_decay` | `[0.0, 0.01, 0.05, 0.1]` | Không decay norm/bias trong custom optimizer nếu phân nhóm |
| `warmup_ratio` | `[0.0, 0.03, 0.06, 0.1]` | Thích hợp theo tổng steps |
| `max_length` | `[128, 256, 512]` | Chỉ nếu model hỗ trợ; kiểm tra actual truncation |
| batch effective | `[16, 32, 64]` | `per_device_batch * grad_accum * num_devices` |
| `dropout` | `[0.0, 0.1, 0.2]` | Cần xác định module đang dùng dropout |
| `seed` | `[13, 42, 2026]` | Với biến động lớn, report mean ± sd |

**Threshold tuning:** binary probability threshold có thể tune trên val; multi-label nên kiểm tra thresholds per-label có regularization/min sample. Không áp dụng threshold tuning cho multiclass softmax tùy tiện.

## 3. NLP embedding / retrieval tuning

- Representation: CLS vs masked mean pooling vs sentence embedding checkpoint; L2 normalize nếu dùng cosine.
- Chunk: 128/256/512 tokens, overlap 0/10–20%, tôn trọng boundary của câu/đoạn.
- Sparse vs dense vs hybrid: BM25/TF-IDF, dense cosine, normalized score fusion hoặc reciprocal-rank fusion.
- Negatives: random vs hard vs in-batch negatives; tránh false negative/same document duplicates.
- Objective: contrastive/infonce/triplet (nếu pipeline train), margin/temperature, batch size, embedding dim.
- Re-ranking: cross-encoder scoring top-k; chi phí tính toán của k, tránh rerank cả corpus.
- Metrics: Recall@k, MRR@k, nDCG@k; latency query, memory index; exact ID matching.
- Order tuning: **data relevance labels** → baseline → checkpoint/pooling → chunk → hybrid → re-rank → supervised contrastive fine-tuning.

## 4. NLP encoder–decoder, translation/summarization

| Category | Hyperparams/technique | Rủi ro |
|---|---|---|
| Training | LR, steps, warmup, label smoothing, gradient clip, teacher forcing theo model | Loss giảm nhưng generation kém |
| Sequence length | `max_source_length`, `max_target_length` | Truncate targets hoặc bỏ facts |
| Decode | greedy vs beams `[1,4,6]`, `length_penalty`, `no_repeat_ngram_size` | Beam rộng tốn latency; output dài vô ích |
| Evaluation | BLEU/chrF, ROUGE, semantic/factual score | Chấm sai do normalization/reference |
| Data | Proper prefix, target language, cleanup | Text leakage, wrong target tokenizer |

## 5. NLP LLM prompt/decode/SFT/LoRA

### 5.1 Trước khi fine-tune

- Baseline zero-shot, few-shot, structured prompt; nhất quán system/user/task instructions.
- Chọn một tập validation có rubric, đánh giá format + correctness + latency + token usage.
- Điều chỉnh `temperature` (0/deterministic cho structured extraction), `top_p`, `max_new_tokens`, few-shot examples/ordering.
- Với API: batch request khi cho phép; cache deterministic outputs nếu hợp lệ; retries có backoff, không lộ key.

### 5.2 Fine-tune / LoRA

- Base/checkpoint và chat template phải khớp; xác định model hỗ trợ `PEFT`/`TRL` hiện hành.
- Data: train kiểu prompt→completion hoặc conversations; check role tags; pack/truncate; `assistant_only_loss` / `completion_only_loss` khi đúng format.
- Methods: full FT; partial FT; LoRA; QLoRA; prompt tuning; adapter tuning (nếu framework hỗ trợ).
- LoRA search: `r=[4,8,16,32,64]`, `alpha` thường tỷ lệ với r (`r`, `2r` là điểm bắt đầu), `dropout=[0,0.05,0.1]`, `target_modules` phải được kiểm tra bằng `named_modules()`, không copy cố định giữa kiến trúc.
- LR search: full SFT thường thử `[1e-5,2e-5,5e-5]`, LoRA thử `[5e-5,1e-4,2e-4]`, tùy model/token budget và loss stability; không coi các số này là luật.
- QLoRA: kiểm tra CUDA/quant backend, `bitsandbytes` compatibility, 4-bit quant type/compute dtype, adapters saving/loading; quantization giảm VRAM nhưng không đảm bảo tăng tốc.
- Compute: gradient checkpointing, bf16/fp16 được GPU hỗ trợ, accumulation, flash attention nếu chính xác và tương thích, freeze embeddings nếu thích hợp.
- Fine-tuning không nhất thiết cần thiết cho bài inference-only/10-minute contest; cân nhắc retrieval, prompt, heuristic trước.

### 5.3 Thuật toán search

1. Manual coarse-to-fine: 3–5 chạy, mỗi lần đổi 1 biến.
2. Random search có không gian rõ, early pruning; chọn theo validation.
3. Optuna Bayesian/TPE khi có đủ ngân sách và repeatable val; không mặc định vượt trội trên 3 runs.
4. Ablation từng thành phần: augmentation only / regularization only / hybrid only.
5. Multi-seed final khi chênh lệch nhỏ, log config hash, model revision.

## 6. Audio classification (HuBERT / Wav2Vec2 / AST tùy dữ liệu)

### 6.1 Các phương án kiến trúc

- Handcrafted spectral features (MFCC/log-mel) + linear/SVM/XGBoost baseline.
- Frozen HuBERT + masked mean/max/statistics pooling + logistic/MLP.
- HuBERT partial unfreeze top-N layers, discriminative LR encoder/head.
- HuBERT full fine-tune; Audio Spectrogram Transformer đối với một số non-speech tasks (đọc model card trước).
- Ensembling checkpoint/augmentation inference nếu latency cho phép.

### 6.2 Search space thí điểm

| Parameter | Candidates | Check |
|---|---|---|
| `max_duration_s` | `[2,5,10]` hoặc thống kê data | Không mất đoạn chứa nhãn |
| `pooling` | masked mean / max / attention | Dùng valid frame mask |
| `encoder_lr` | `[1e-6, 3e-6, 1e-5]` (khi unfreeze) | Kiểm tra instability |
| `head_lr` | `[1e-4, 3e-4, 1e-3]` | Head-only cần lịch khác FT |
| `unfreeze` | `[0, last_2, last_4, all]` | Params/VRAM và time |
| `dropout` | `[0.1,0.3,0.5]` | Overfit vs underfit |
| aug intensity | none/light/moderate | Ablation trên clean/noisy val |

**Lưu ý quan trọng:** sample rate phải khớp yêu cầu checkpoint; HuBERT không tự động có nghĩa là mọi checkpoint chịu được mọi sample rate.

## 7. Whisper ASR tuning

### 7.1 Dữ liệu là ưu tiên số 1

- Đúng waveform/audio-text alignment, clip length, sample rate, labels, speaker split.
- Kiểm tra transcript: punctuation/diacritics/digits/language conventions, tags `[noise]`, `[laugh]` có được tính vào WER hay không.
- Nếu audio dài, segmentation + chunk overlap + safe merge; bảo vệ nguyên câu.
- Language ID/forced language option; tránh model auto-detect nhầm trong audio ngắn/code-switch.

### 7.2 Training candidates

- `lr`: thử `[5e-6,1e-5,3e-5]`, `warmup`, `max_steps`, `batch_size`, `gradient_accum`, mixed precision, checkpointing.
- Fine-tuning nhiều bước không luôn cải thiện; val WER / per-language WER là tiêu chí.
- Freeze encoder / partial unfreeze hoặc LoRA *nếu version/model hỗ trợ*; đánh giá chất lượng so với full FT.
- Pseudo-label: chỉ nhận confident subset, có risk confirmation bias; đừng đem validation/test vào train.

### 7.3 Inference/decoding candidates

- `beam_size` 1/3/5, model size tiny→small→medium, language/task selection, VAD on/off, temperature fallback khi wrapper hỗ trợ.
- Chunk duration/overlap; VAD aggressiveness; timestamps/segment merging; hallucination guard.
- Với `faster-whisper` (CTranslate2): benchmark int8/float16 vs float32 nếu hardware hỗ trợ, batch vs single inference, compute type; **không** giả định mọi checkpoint convert được không lỗi.
- Metric triple: **WER + RTF + peak VRAM**; nếu loại ASR hỗ trợ, timestamps error riêng.

## 8. Qwen-Audio / Voxtral inference tuning

- Model version khả năng riêng: Qwen-Audio (gốc) / Qwen2-Audio / Qwen2.5-Omni; Voxtral local checkpoint/API versions.
- 1) Prompt/task framing rõ (transcribe vs audio QA vs caption); 2) model-specific chat template/processor; 3) max audio length; 4) batching/chunks; 5) decode length/temperature; 6) audio-only vs transcript+LLM cascade.
- Tune output validation: exact match/structured parse rate, factual support in audio, hallucination rate trên silent/noise clip.
- Nếu model lớn hơn GPU: thử smaller variant, quantization **nếu được hỗ trợ**, CPU offload với lưu ý latency, model API trước thi nếu được phép. Không khẳng định local fine-tuning nhẹ.
- Model API thay đổi: gắn link docs + model ID + date + code version trong config. Không giả định Voxtral model mới nhất dùng API giống bản cũ.

## 9. Tuning hiệu năng / phần cứng

| Kỹ thuật | Hiệu quả kỳ vọng | Bẫy |
|---|---|---|
| Mixed precision bf16/fp16 | Tiết kiệm VRAM/tăng throughput tùy GPU | fp16 overflow / không hỗ trợ dtype |
| Gradient accumulation | Effective batch lớn hơn batch vật lý | Throughput giảm; optimizer steps thay đổi |
| Gradient checkpointing | Giảm activation memory | Tốn compute thêm |
| Dynamic padding / length bucketing | Giảm token/frame pad và waste | Không đổi ngầm loss weighting |
| Freeze encoder | Giảm gradient/optimizer memory | Giảm adaptation |
| Quantized inference | Giảm footprint tùy backend | Accuracy/latency/compatibility |
| Precompute embeddings | Train classifier nhanh | Disk/memory; không valid nếu encoder đang FT |
| Cache/preload | Giảm IO | Vượt RAM/disk; bất hợp lệ khi không cho phép |
| Batch inference | Tăng throughput | OOM; latency từng request |
| Distillation | Nhẹ hơn khi deploy | Phải có train data và teacher outputs hợp pháp |

**Dừng khi:** benchmark không cải thiện rõ, runtime quá ngân sách, VRAM vượt hạn, hoặc còn rủi ro format/inference offline. Một baseline đơn giản nhưng chạy chắc thường có giá trị hơn FT rất chậm.

## 10. Bảng quyết định xử lý lỗi nhanh

| Symptom | Investigate first | Next action |
|---|---|---|
| Loss = NaN | FP16 overflow, LR, bad labels, all-masked batch | giảm LR, bf16/float32, grad clip, inspect batch |
| CUDA OOM | Batch/length, cache, model size, precision | giảm length/batch, accumulation, checkpointing |
| Train good / val bad | Leakage, overfit, distribution shift | group split, early stop, wd, data audit |
| Train+val poor | wrong model/task, normalization, label map | fix data/processor/learning setup |
| Text output gibberish | wrong tokenizer/template, decode config | verify model card/template, special tokens |
| ASR random hallucination | silence/noise, language, long context/chunks | VAD, language config, decode thresholds |
| Audio model gets no speech | wrong sample rate/channels/processor | inspect waveform, resample genuinely |
| Val improves but inference fails | artifacts missing, overfitted format assumptions | reload from saved dir, smoke test offline |

## 11. Experiment tracking schema

Mọi thí nghiệm ghi `run_id`, `task`, `dataset/version/hash`, `split/seed`, `model/revision`, `processor/tokenizer`, `trainable_params`, `hyperparameters`, `metric_name`, `metric_value`, `val_examples`, `gpu/hardware`, `peak_vram_mb`, `train_seconds`, `inference_seconds`, `notes`. Dùng `experiments.csv` + `metrics.json`, không bịa kết quả.

## 12. Tối ưu theo giới hạn contest 5–12 phút

**0–10% thời gian:** đọc input/output, giới hạn, scorer, baseline và format. **10–30%:** run baseline + validate local. **30–80%:** thử một thay đổi nhiều kỳ vọng nhất (features, text coherence, correct decode/model) và so scorer. **80–100%:** freeze solution, smoke test trên edge cases, fallback khi fail. Tỷ lệ chỉ là gợi ý quản lý thời gian, **không** phải lịch thi chính thức.
