# NLP & Audio — IOAI/UET 2026 (ALL-IN-ONE)
> Tệp tổng hợp tiện gửi cho AI khi không muốn attach nhiều file. Các tệp Markdown riêng vẫn dễ cập nhật hơn.



---

<!-- SOURCE_FILE: README.md -->

# IOAI / OlpAI UET 2026 — NLP & Audio Practical Handbook

> Phiên bản: 2026-10-23 · Ngôn ngữ: tiếng Việt · Mục đích: tài liệu ôn thực hành và **đặc tả để đưa cho AI sinh code**.
>
> Lưu ý: đây là lộ trình đề xuất dựa trên phạm vi đã cung cấp và IOAI 2026 syllabus, **không phải đề thi dự đoán**. Các giá trị hyperparameter là *điểm khởi đầu*, không phải giá trị tối ưu phổ quát.

## 1. Cách sử dụng

- **Học viên:** mở `00_SKILL_MATRIX.md`, học theo `03_LAB_ROADMAP.md`, thực hành ở `01_NLP_PRACTICE.md` / `02_AUDIO_PRACTICE.md` và đối chiếu `05_EVALUATION_DEBUG.md`.
- **Người chuẩn bị bài:** lấy một `LAB-ID`, gửi nội dung file tương ứng + `06_AI_CODEGEN_PROMPTS.md` + YAML trong `configs/` cho AI để sinh notebook, bài tập và grader.
- **Khi muốn cải thiện điểm:** đọc `04_TUNING_PLAYBOOK.md` theo đúng loại task; dùng thử nghiệm có kiểm soát trên validation, không tối ưu trên test.
- **Khi cần mẫu code:** xem `07_REPOS_RESOURCES.md`; ưu tiên repo chính thức, notebook và tài liệu với API phiên bản hiện hành.

## 2. Cấu trúc

| File | Nội dung | Dùng khi |
|---|---|---|
| `00_SKILL_MATRIX.md` | Toàn bộ kỹ năng, mức độ, đầu ra mong đợi, tiêu chí thành thạo | Lập checklist / chia phần học |
| `01_NLP_PRACTICE.md` | BERT, text classification, embeddings, encoder-decoder, language modeling, LLM/API | Tạo lab NLP |
| `02_AUDIO_PRACTICE.md` | Waveform, resampling, HuBERT, Whisper, Qwen-Audio, Voxtral | Tạo lab Audio |
| `03_LAB_ROADMAP.md` | 6 tuần, 12 lab và 3 mini-contest, bài IOAI gốc | Lên chương trình đào tạo |
| `04_TUNING_PLAYBOOK.md` | Data, training, LoRA/QLoRA, decoding, inference, search strategy, chống OOM | Cải thiện mô hình |
| `05_EVALUATION_DEBUG.md` | Metrics, lỗi phổ biến, reproducibility, leakage, protocol so sánh | Chấm và debug |
| `06_AI_CODEGEN_PROMPTS.md` | Prompt generator cho AI + hợp đồng đầu ra + 6 prompt task cụ thể | Sinh code bằng AI |
| `07_REPOS_RESOURCES.md` | GitHub / docs chính thức, repo IOAI theo bài | Tra cứu code |
| `configs/*.yaml` | Các preset cấu hình minh họa để AI biến thành code | Seed thông số |

## 3. Các năng lực cốt lõi cần đạt

1. Từ dữ liệu thô → EDA → chia split chống rò rỉ → preprocessing → baseline → train/eval → inference → báo cáo.
2. **NLP:** TF-IDF + linear model; BERT encoder; embeddings/cosine; T5/BART encoder-decoder; Causal LM/API; LoRA.
3. **Audio:** kiểm tra sample rate, kênh, độ dài, tín hiệu; resampling + crop/pad; HuBERT features/fine-tuning; Whisper ASR + WER; Qwen-Audio/Voxtral audio understanding.
4. Tuning có giả thuyết: đo baseline, thay **một** yếu tố, chạy có seed, ghi số đo và trade-off tài nguyên.
5. Làm bài thời gian giới hạn: kiểm tra format nộp, chạy smoke test, giữ baseline dự phòng, bảo đảm inference offline nếu đề cấm Internet.

## 4. Phân biệt mức độ tài liệu

- **[SYLLABUS]**: được nêu trong syllabus IOAI 2026 hoặc đề cương UET người học cung cấp.
- **[EXTENSION]**: đề xuất bổ sung để giải bài nhanh/chắc, không khẳng định là nội dung IOAI bắt buộc (ví dụ TF-IDF, VAD, CER, SpecAugment, LoRA, validation theo speaker).
- **[REFERENCE]**: mã nguồn hoặc tài liệu bên ngoài; kiểm tra API, model card, license, dataset trước khi sử dụng.

## 5. Nguồn gốc

- Official syllabus: https://ioai-official.org/wp-content/uploads/2025/10/Syllabus.pdf
- IOAI 2026: https://github.com/IOAI-official/IOAI-2026
- IOAI 2025: https://github.com/IOAI-official/IOAI-2025
- IOAI 2024: https://github.com/IOAI-official/IOAI-2024
- Official resources: https://ioai-official.org/resources/

**Ghi chú IOAI 2026**: README chính thức mô tả phiên thi cá nhân trên notebook, một GPU khoảng 16 GB, không có truy cập Internet và thời gian 5–12 phút mỗi bài. Vì vậy *không nên* xem fine-tuning lâu là mặc định; pre-trained inference, feature extraction, heuristic, retrieval và tối ưu giải thuật có thể thực tế hơn. Phân biệt `code/*-original` (artifact thi chính thức) với `code/baseline` và `code/grading` (bản educational ngoài môi trường thi).

## 6. Quy tắc cứng khi AI sinh code từ bộ này

- Không được giả định một API/model ID chạy được nếu chưa xác nhận version/hardware/access trong môi trường.
- Không tải checkpoints/data khi chạy bài offline; phải chuẩn bị và kiểm tra cache trước.
- Không dùng tập test để chọn siêu tham số; validation phải giữ riêng và loại leakage theo nguồn/speaker nếu thích hợp.
- Không được gọi cấu hình YAML là đã thử nghiệm. Chỉ ghi `RESULT` khi thực sự có output đo được.
- Trả về `requirements.txt`, notebook hoặc Python scripts, config, README cách chạy, seed, metrics, file kết quả và sanity checks.


---

<!-- SOURCE_FILE: 00_SKILL_MATRIX.md -->

# Skill Matrix — NLP & Audio và kỹ năng hỗ trợ

## Ký hiệu

- **Cả hai (Theory+Practice):** giải thích được nguyên lý/mục đích và triển khai bằng thư viện.
- **Thực hành:** dùng được, nhận diện khi áp dụng, xử lý dữ liệu và giải thích đầu ra.
- **P0:** phải thành thạo; **P1:** quan trọng; **P2:** tăng cường/đào sâu.
- **Pass:** có code chạy + đánh giá đúng + giải thích lý do lựa chọn. Không chỉ chạy thành công một notebook tải sẵn.

## A. NLP — theo phạm vi 3.4

| ID | Chủ đề | Mức | Ưu tiên | Bài kiểm tra năng lực (Deliverable) |
|---|---|---|---|---|
| NLP-01 | Tokenization, vocabulary, attention mask, truncation, dynamic padding | Thực hành, liên quan 3.1 | P0 | Tokenize 10 mẫu, in token/ID/mask, kiểm tra shape và padding |
| NLP-02 | Pretrained text encoder: BERT/DistilBERT | Cả hai | P0 | Load tokenizer/model, trích embedding có mask pooling, so sánh cosine |
| NLP-03 | Text classification (binary, multiclass, multilabel) | Thực hành | P0 | TF-IDF baseline + fine-tune encoder, macro-F1/confusion matrix |
| NLP-04 | Language modeling: masked vs causal LM; next-token loss/perplexity | Cả hai | P1 | So sánh dự đoán token/masking, tính loss/perplexity với dữ liệu phù hợp |
| NLP-05 | Encoder–decoder: T5/BART, translation/summarization | Thực hành | P0 | DataCollatorForSeq2Seq, fine-tune/infer, BLEU/chrF hoặc ROUGE |
| NLP-06 | Pretrained LLM open-source: prompts, chat template, batching, generation | Thực hành | P0 | Chạy inference nhiều mẫu; đo accuracy/quality + latency |
| NLP-07 | LLM API: messages, response parsing, JSON/schema, retries, rate limits | Thực hành | P1 | Client dùng biến môi trường; log error, không hard-code key |
| NLP-08 | Full fine-tuning vs LoRA/QLoRA, frozen encoder, linear probing | 3.2 Thực hành | P0 | Báo cáo trainable params, VRAM, metric; ít nhất 2 chế độ |
| NLP-09 | Embedding retrieval, nearest-neighbor, semantic similarity | Bổ trợ [EXTENSION] | P1 | Recall@k/MRR@k trên tập retrieval, chống duplicate leakage |
| NLP-10 | Text augmentation / handling class imbalance | Bổ trợ [EXTENSION] | P1 | Ablation cho class weights, sampling, augmentation an toàn |

## B. Audio — theo phạm vi 3.4

| ID | Chủ đề | Mức | Ưu tiên | Bài kiểm tra năng lực |
|---|---|---|---|---|
| AUD-01 | Waveform, sample rate, mono/stereo, amplitude, clip, duration | 3.1 + Audio Thực hành | P0 | Load WAV/MP3; đo duration/sample rate; nghe/plot kiểm tra |
| AUD-02 | Resample, crop, padding, attention mask/length, batch data collator | 3.1 + Audio Thực hành | P0 | Chạy batch audio độ dài khác nhau; kiểm tra seconds/tensor shape |
| AUD-03 | HuBERT pretrained encoder | Thực hành | P0 | Load HuBERT feature extractor/encoder; mean pooling có length mask |
| AUD-04 | Fine-tuning HuBERT audio classification | Thực hành | P0 | Linear probe vs partial/full unfreeze; macro-F1; split theo speaker |
| AUD-05 | Whisper ASR inference | Thực hành | P0 | Audio → transcript, đúng language/task, benchmark WER/CER |
| AUD-06 | Whisper fine-tuning và decoding | Thực hành | P0 | Seq2seq ASR; ablation decode; lỗi theo thời lượng/noise |
| AUD-07 | Qwen-Audio audio–text understanding | Thực hành | P1 | Hỏi đáp/caption/nhận diện sự kiện từ audio; theo đúng template model |
| AUD-08 | Voxtral audio processing | Thực hành | P1 | ASR hoặc audio QA theo *model/version* được cung cấp; thống kê latency |
| AUD-09 | VAD/chunking, silence trimming, loudness/noise augmentation | Bổ trợ [EXTENSION] | P1 | Benchmark latency và WER trước/sau preprocessing |
| AUD-10 | Speaker-aware, source-aware/group-aware validation | Bổ trợ [EXTENSION] | P0 | Không có cùng speaker/session giữa train/test nếu yêu cầu tổng quát hóa |
| AUD-11 | Speech → text → NLP downstream pipeline | Bổ trợ [EXTENSION] | P1 | So sánh dự đoán từ transcript ASR và gold transcript |

## C. Kỹ năng nền tảng hỗ trợ cả NLP và Audio

| ID | Kỹ năng | Phải làm được |
|---|---|---|
| GEN-01 | Python, NumPy/pandas, matplotlib | Đọc bảng và file; EDA nhãn/length, vẽ distribution và error cases |
| GEN-02 | PyTorch tensor, DataLoader, GPU | `to(device)`, train/eval/no_grad, mixed precision đúng phần cứng, batch/shape |
| GEN-03 | Loss + optimizer | Cross-entropy, BCEWithLogits, CTC / seq2seq loss tùy task; AdamW, scheduler |
| GEN-04 | Metrics | Accuracy, macro/micro/weighted F1, P/R, confusion matrix, WER/CER, BLEU/ROUGE theo task |
| GEN-05 | Train/val/test | Stratification / group split, seed, validation metric, checkpoint best |
| GEN-06 | Overfitting, early stopping | Đọc train/val curves và can thiệp phù hợp |
| GEN-07 | Tuning | Có log hyperparameter, ablation, ghi runtime/VRAM; không cherry-pick test |
| GEN-08 | Reproducibility | Seed, model revision/version, config, data manifest, saved outputs |
| GEN-09 | Offline reproducibility | Model + tokenizer/processor + data local; không HTTP; chạy lại sau restart |
| GEN-10 | Submission engineering | Đúng format grader, smoke test, time/VRAM budget, fallback baseline |

## D. Đánh giá thành thạo

1. **Level 0 — Nhận diện:** biết input/output model và chọn đúng task.
2. **Level 1 — Inference:** chạy pre-trained, xử lý edge cases, có metric.
3. **Level 2 — Training:** viết pipeline dataset → loss → eval → checkpoint.
4. **Level 3 — Tuning:** tăng kết quả validation bằng 3 phép thử có kiểm soát và báo cáo ablation.
5. **Level 4 — Competition-ready:** chạy offline, không rò rỉ, đúng định dạng đầu ra, đạt giới hạn bộ nhớ/thời gian.

## E. Gợi ý phân công học

- **NLP owner:** NLP-01 đến NLP-10, chịu trách nhiệm 6 lab NLP và 1 mini-contest text.
- **Audio owner:** AUD-01 đến AUD-11, chịu trách nhiệm 6 lab Audio và 1 mini-contest speech/audio.
- **Cùng phụ trách:** GEN-01 đến GEN-10, đặc biệt evaluation, split, LoRA và benchmark compute.

Đặc biệt: syllabus chỉ nêu **HuBERT, Whisper, Qwen-Audio, Voxtral** ở mức thực hành; **VAD, SpecAugment, CTC, LoRA cho audio** là kỹ thuật mở rộng hữu ích chứ không phải tất cả đều được nêu riêng trong syllabus.


---

<!-- SOURCE_FILE: 01_NLP_PRACTICE.md -->

# NLP — Học và thực hành theo dạng bài

> Mỗi module cần có notebook/script, dataset schema, metrics, baseline, tuning và phân tích lỗi. Phân biệt minh họa từ dữ liệu thực nghiệm.

## NLP-01 — Tokenization & dataset pipeline (P0)

### Thành thạo

- Từ raw text → label map → train/val/test (stratified khi hợp lý) → tokenizer → attention_mask → collator.
- Hiểu WordPiece/BPE/SentencePiece ở mức sử dụng; token ID không phải từ; Unicode, tiếng Việt, emoji và ngôn ngữ hỗn hợp làm thay đổi token count.
- Tránh *fit* TF-IDF, vocabulary hoặc normalize statistics trên cả validation/test.
- Dynamic padding theo batch; truncation cắt bỏ phần quan trọng; `max_length` phải dựa vào độ dài và kiến trúc.

### Thực hành bắt buộc

1. Thống kê p50/p90/p95/p99 số token và tỷ lệ bị truncate.
2. So sánh fixed padding vs dynamic padding (latency và token padding ratio).
3. Kiểm tra data có near duplicates, nhãn không cân bằng, null/empty texts.

## NLP-02 — Text classification: baseline → BERT encoder (P0)

### Quy trình

1. Baseline `TfidfVectorizer` + `LogisticRegression` (có thể thử char n-grams cho tiếng Việt hoặc dữ liệu noisy).
2. Pretrained checkpoint phù hợp ngôn ngữ; `AutoTokenizer`, `AutoModelForSequenceClassification`, dynamic collator.
3. Loss: CE cho single-label; BCEWithLogits cho multi-label; không dùng softmax+suy luận argmax cho multi-label.
4. Validation: macro-F1 chính nếu imbalance; thêm per-class P/R/F1 và confusion matrix.
5. Tune learning rate, batch/accumulation, epochs, max_length, weight decay, scheduler, warmup, class weights, freeze/unfreeze.

### Bài lab (`LAB-NLP-01`)

- Dataset gợi ý: `stanfordnlp/imdb` (binary, tiếng Anh) hoặc dữ liệu phân loại được phép sử dụng.
- Baseline 1: TF-IDF + LogisticRegression; baseline 2: pre-trained BERT/DistilBERT frozen + linear head; baseline 3: fine-tune.
- Output: `metrics.json`, confusion matrix PNG, error analysis CSV tối thiểu 20 dự đoán sai, checkpoint.
- Tuning hợp lệ: LR `[1e-5, 2e-5, 3e-5, 5e-5]`, epochs `[2,3,5]`, length `[128,256,512]` (nếu model hỗ trợ), batch `[8,16]`.
- Exit: inference trên text chưa thấy; label names đúng; best checkpoint chọn bằng val F1.

## NLP-03 — Pretrained encoders & embeddings (P0/P1)

- `AutoModel`: `last_hidden_state` shape `[B, T, H]`; CLS vs mean pooling **theo attention mask** vs max pooling.
- Normalize vector trước cosine/dot product khi mục tiêu cosine; không luôn giả định CLS tốt cho similarity.
- Với semantic similarity/retrieval: cân nhắc checkpoint sentence embedding được đào tạo đúng mục tiêu; so với base BERT.
- Metrics: Spearman/Pearson cho pair similarity; Recall@k/MRR@k/nDCG@k cho retrieval có relevance labels.
- Bài `LAB-NLP-02`: cho bộ câu hỏi và tài liệu, retrieval bằng TF-IDF/BM25 so với dense encoder; benchmark CPU/GPU và memory.

## NLP-04 — Language modeling / Causal LM (P1)

- Phân biệt BERT masked LM, GPT-style causal LM, encoder–decoder conditional LM.
- LM training: tokenization, causal mask, shifted labels, `-100` để bỏ qua token không tính loss; trainable data vs eval split.
- Perplexity chỉ so sánh có ý nghĩa với **cùng tokenizer, phân phối và cách tính loss**, không áp dụng bừa bãi cho chatbot.
- Inference: greedy, beam (nếu thích hợp), sampling (`temperature`, `top_p`, `top_k`), max_new_tokens, stop tokens; determinism khi cần chấm tự động.
- `LAB-NLP-03`: prompt 20 yêu cầu, so sánh decoding về chất lượng/task accuracy, repetition, latency, format adherence.

## NLP-05 — Encoder–decoder (P0)

- T5/BART (nếu đúng checkpoint), `AutoModelForSeq2SeqLM`, `DataCollatorForSeq2Seq`, tách input/target.
- Translation: BLEU/chrF (chuẩn tokenization/ngôn ngữ); Summarization: ROUGE + đánh giá factuality thủ công hoặc bằng rubric.
- Input prompt/prefix theo model; attention mask và target labels; label padding dùng `-100`.
- Decoding: beam_width, length_penalty, no_repeat_ngram_size, max_new_tokens.
- `LAB-NLP-04`: translation (ví dụ OPUS Books theo HF tutorial) hoặc summarization, train ít bước rồi ablate beam=1/4; báo cáo metric, latency, độ dài output.

## NLP-06 — Pretrained LLM: local + API (P0)

### Local (open-source/weights available)

- Đọc model card, tokenizer, chat template, license, VRAM, context length, special tokens.
- API inference của model phải đúng version; ưu tiên `apply_chat_template` từ tokenizer/processor thay vì tự nối raw tags.
- Dùng `torch.no_grad()`/inference mode, `eval()`, batched inference, max_new_tokens và stop condition.
- Bài `LAB-NLP-05`: benchmark 3 kiểu prompt (zero/few-shot/structured) trên cùng tập validation, chấm câu trả lời chuẩn hóa (exact match/accuracy, parsing success rate).

### API

- Không hard-code API key; lấy từ `os.environ`.
- Xử lý timeout, 429, 5xx, exponential backoff có giới hạn, logging không chứa dữ liệu nhạy cảm.
- Ghi model ID/version, prompt template, token usage, latency, số lần fail, giá nếu có số liệu thực.
- JSON schema/format compliance phải validate bằng code; kết quả bất hợp lệ = lỗi cần ghi log, không tự gán đúng.
- Nơi đề cấm internet: API chỉ là chủ đề thực hành trước khi thi; không coi API là phương án thi hợp lệ.

## NLP-07 — Full FT, frozen encoder, LoRA/QLoRA (P0)

| Phương án | Cập nhật | Dùng khi | Rủi ro |
|---|---|---|---|
| Zero/few-shot | Không | Không có thời gian train | Prompt sensitivity |
| Frozen embeddings + head | Head nhỏ | Dữ liệu ít, compute yếu | Domain gap |
| Partial unfreeze | Một số blocks + head | Có giới hạn GPU | Cần lựa layer |
| Full fine-tuning | Toàn bộ model | Model vừa, đủ GPU/dữ liệu | OOM/overfit |
| LoRA | Adapter low-rank | Model lớn, cần giảm memory | Chọn target modules/learning rate |
| QLoRA | Base thường 4-bit + LoRA | GPU hạn chế, phần cứng hỗ trợ | Quantization compatibility/overhead |

Thực hiện so sánh có công bằng: cùng split, metric, data budget, train steps (trong mức khả thi), ghi trainable params/VRAM/latency.

### Lab `LAB-NLP-06`

- Instruction dataset nhỏ hợp pháp, conversational/prompt-completion chuẩn, baseline inference → LoRA SFT (`trl` + `peft`).
- `trainable_parameters`, `train_loss/val_loss`, format accuracy, val task accuracy, train runtime, peak VRAM.
- Lưu adapter + base model ID/revision + tokenizer, thử reload rồi infer độc lập.

## NLP-08 — Tuning theo bản chất dữ liệu

| Triệu chứng | Chẩn đoán | Biện pháp thử |
|---|---|---|
| Train F1 cao, val F1 thấp | Overfit hoặc leakage phía train | Weight decay, dropout, augmentation hợp lệ, early stop, group split |
| Macro-F1 thấp, accuracy cao | Class imbalance | Class weights, stratified split, threshold tuning trên val (binary/multi-label) |
| Văn bản dài, truncate nhiều | Thiếu ngữ cảnh | Max length vừa đủ, sliding window + aggregation, long-context checkpoint |
| Lỗi tên riêng/tiếng Việt | Domain/tokenizer mismatch | Multilingual/vi pretrained checkpoint; normalization nhẹ và ablation |
| Kém semantic matching | CLS embedding không đúng mục tiêu | Sentence embedding checkpoint, masked mean pooling, similarity objective |
| Sinh output sai định dạng | Decode/prompt/stop không chắc | Schema, deterministic decode, constrained parsing, validation |

## NLP-09 — Bài IOAI gốc để ôn

- IOAI 2025 `Concepts`: bài liên quan LLM/API, có baseline, solution và `judge_api.py` — xem README và giới hạn chạy lại ngoài môi trường thi.
- IOAI 2026 `Potato`: semantic word guessing, thử word embeddings/similarity/search dưới giới hạn lượt.
- IOAI 2026 `Ghost of the Machine`: xác định chỉnh sửa do máy tạo trong các đoạn sách; thích hợp ôn tokenization, sequence scoring, classifier và error localization.
- IOAI 2026 `Robot Chasing`: chuyển chỉ dẫn ngôn ngữ tự nhiên thành hành vi trong môi trường gridworld; kết hợp parsing/LLM với logic.

Nguồn code chính thức: `07_REPOS_RESOURCES.md`.


---

<!-- SOURCE_FILE: 02_AUDIO_PRACTICE.md -->

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


---

<!-- SOURCE_FILE: 03_LAB_ROADMAP.md -->

# Lộ trình ôn tập 6 tuần — 12 lab + 3 mini-contest

> Giả định người học đã biết Python/PyTorch cơ bản. Có thể nén thành 3 tuần nếu ghép hai tuần liên tiếp; tăng thời lượng nếu chưa quen training. Mỗi lab yêu cầu 1 notebook + 1 README + 1 metrics.json + ít nhất 1 lần tuning/ablation.

## Tuần 1 — Nền tảng pipeline

| Lab | Lĩnh vực | Nhiệm vụ | Đạt khi |
|---|---|---|---|
| LAB-NLP-01 | NLP | IMDB/nhãn văn bản: TF-IDF LogisticRegression → encoder fine-tune | Đúng train/val/test; macro-F1/confusion; lưu best model |
| LAB-AUD-01 | Audio | Raw audio EDA, resample, plotting, duration, normalization | Load 3 sample rates; padding/collator; validate duration |

**Mini-contest MC-1:** cung cấp bộ classification text với label imbalance và giới hạn VRAM. Chấm macro-F1 + đúng format + không lệ thuộc Internet ở inference.

## Tuần 2 — Representation learning

| Lab | Lĩnh vực | Nhiệm vụ | Đạt khi |
|---|---|---|---|
| LAB-NLP-02 | NLP | Sentence embeddings và retrieval; dense vs lexical | Recall@5/MRR@5; per-query errors; cosine đúng |
| LAB-AUD-02 | Audio | HuBERT: frozen features+head vs partial fine-tune | Macro-F1; resource logs; speaker split nếu thích hợp |

## Tuần 3 — Sequence / Generation

| Lab | Lĩnh vực | Nhiệm vụ | Đạt khi |
|---|---|---|---|
| LAB-NLP-03 | NLP | Causal LM + prompt/generation decode study | 3 decoding settings + metrics + output parse rate |
| LAB-AUD-03 | Audio | Whisper ASR zero-shot, normalize, WER/CER | Transcript errors phân nhóm; RTF/memory |

**Mini-contest MC-2:** data audio bị nhiễu và length shift. Chấm WER hoặc audio classification macro-F1 tùy bộ được chọn; giới hạn latency.

## Tuần 4 — Encoder–decoder và ASR fine-tune

| Lab | Lĩnh vực | Nhiệm vụ | Đạt khi |
|---|---|---|---|
| LAB-NLP-04 | NLP | Encoder–decoder translation/summarization | BLEU/chrF hoặc ROUGE; model reload; decode ablation |
| LAB-AUD-04 | Audio | Fine-tune Whisper bằng Seq2SeqTrainer | Baseline vs FT WER; proper padding labels, reproducibility |

## Tuần 5 — LLM và Audio-LLM

| Lab | Lĩnh vực | Nhiệm vụ | Đạt khi |
|---|---|---|---|
| LAB-NLP-05 | NLP | Local/API LLM; zero/few-shot; structured output | Valid JSON/format; latency; API key safe; no leak |
| LAB-AUD-05 | Audio | Qwen-Audio/Qwen2-Audio: audio QA/caption | 10–30 clip; task accuracy rubric; hallucination log |

## Tuần 6 — Tuning và thi thử

| Lab | Lĩnh vực | Nhiệm vụ | Đạt khi |
|---|---|---|---|
| LAB-NLP-06 | NLP | LoRA SFT với PEFT/TRL; compare zero-shot | Trainable params, best val, memory, reload adapter |
| LAB-AUD-06 | Audio | Voxtral vs Whisper hoặc mô hình audio khác tương thích | WER/quality/RTF công bằng + phân tích điểm yếu |

**Mini-contest MC-3:** tái hiện bài `Find the Order` IOAI 2026 trên educational baseline và grader. Có thể thêm `Potato` hoặc `Ghost of the Machine` nếu nhóm NLP tiến nhanh.

## Bài IOAI thật được ưu tiên (mức độ phù hợp)

| Bài | Năm | Lĩnh vực | Làm gì |
|---|---:|---|---|
| [Find the Order](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/1_Find_the_Order) | 2026 | Audio+NLP | ASR rồi sắp xếp dialogue turns; có baseline/eval educational |
| [Potato](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/3_Potato) | 2026 | NLP | Semantic similarity/search, thử chiến lược word guessing |
| [Ghost of the Machine](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/5_Ghost_of_Machine) | 2026 | NLP | Phát hiện đoạn văn bị máy chỉnh sửa |
| [Robot Chasing](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/2_Robot_Chasing) | 2026 | NLP/reasoning | Hiểu chỉ dẫn ngôn ngữ tự nhiên trong gridworld |
| [Concepts](https://github.com/IOAI-official/IOAI-2025/tree/main/Individual-Contest/Concepts) | 2025 | LLM/API | Baseline, lời giải và simulated judge API; cần key/token ngoài môi trường thi |

## Chuẩn output mỗi lab

```
lab_id/
├── README.md                 # Mục tiêu, dataset, setup, data split, cách chạy
├── solution.ipynb            # Notebook giải thích + thực hành
├── config.yaml               # Seed/hyperparameter/data paths
├── requirements.txt          # Dependencies phiên bản thực tế đã chạy
├── outputs/metrics.json      # Chỉ dữ liệu ĐÃ CHẠY
├── outputs/predictions.csv   # id,prediction (và optional confidence)
├── outputs/error_analysis.csv
└── outputs/experiments.csv   # Baseline / ablation / runtime
```

## Cách chấm chung (gợi ý, không phải IOAI official rubric)

- 35% đúng dữ liệu, split, pipeline và evaluation.
- 25% chất lượng validation so với baseline.
- 20% tuning có căn cứ + error analysis.
- 15% reproducibility, thời gian/VRAM, inference offline nếu yêu cầu.
- 5% báo cáo rõ ràng và code dễ tái sử dụng.

Ưu tiên thành thạo 2–3 pipeline từ đầu đến cuối hơn chạy qua hàng chục checkpoint mà không biết vì sao khác điểm.


---

<!-- SOURCE_FILE: 04_TUNING_PLAYBOOK.md -->

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


---

<!-- SOURCE_FILE: 05_EVALUATION_DEBUG.md -->

# Evaluation, anti-leakage và debug checklist

## 1. Phải khai báo trước khi chạy

```yaml
task_type: text_classification    # asr/audio_classification/retrieval/translation/generation
primary_metric: macro_f1
maximize: true
split_strategy: stratified         # group_speaker/group_recording/temporal/official
seed: 42
dataset_revision: record_exact_version
model_revision: record_exact_revision
validation_policy: tune_only_on_validation
test_policy: final_report_only
```

## 2. Chọn metric đúng

| Task | Primary suggested | Secondary | Không nên làm |
|---|---|---|---|
| Binary classification | F1 (nếu imbalance) hoặc AUROC tùy goal | precision/recall, PR-AUC, confusion | Chỉ report accuracy trên 99:1 data |
| Multiclass classification | Macro-F1 | Weighted-F1, accuracy, per-class | Chỉ dùng micro-F1 khi các lớp lệch |
| Multi-label classification | Macro/micro-F1 theo goal | exact match, per-label PR-AUC | Argmax mọi nhãn |
| Regression | MAE/RMSE | R² | Dùng accuracy |
| Embedding similarity | Spearman | Pearson | Dùng classification accuracy không định nghĩa |
| Retrieval | MRR@k / Recall@k | nDCG@k | Index chứa positive duplicates từ test |
| ASR | WER | CER, RTF, subgroup WER | So raw vs normalized một cách không nhất quán |
| Translation | chrF/BLEU | human adequacy | So metric giữa normalization khác |
| Summarization | ROUGE + human factuality | length/compression | Tin ROUGE phản ánh factuality hoàn toàn |
| Audio QA | task accuracy / rubric | grounding/hallucination, latency | Chấm text generation chỉ bằng WER |

**WER** = (substitutions + deletions + insertions) / words in reference; **CER** tương tự ở mức ký tự. Nếu ngôn ngữ không tách từ bằng khoảng trắng, WER phụ thuộc tokenizer; có thể ưu tiên CER/segmentation rõ.

## 3. Data leakage checklist

- [ ] Không fit tokenizer mới, vectorizer, feature scaler, PCA trên test; pretrained tokenizer có sẵn là trường hợp khác.
- [ ] Không oversample hoặc augment trước split (near-duplicates sẽ xuyên tập).
- [ ] Không có cùng document/user/conversation/speaker/session giữa train/test nếu đánh giá tổng quát hóa theo group.
- [ ] Loại near-duplicate text/waveform/clip overlap nếu tương tác với target.
- [ ] Không dùng test predictions/public private labels để fine-tune khi luật không cho.
- [ ] Hyperparameter và threshold chọn bằng validation, không dùng test.

## 4. Audio-specific diagnostics

- [ ] Audio decodes, sample rate is correct and truly resampled.
- [ ] Speech/noise duration distribution; no pathological empty waveform.
- [ ] Grouping by speaker/recording if required.
- [ ] Audio peak/clipping, silence, corrupt channels.
- [ ] Processor and model match (e.g., sample rate, language/task, mel bins).
- [ ] Variable-length padding/attention lengths correct after feature downsampling.
- [ ] ASR labels pad positions are ignored with `-100`; decode excludes special tokens correctly.
- [ ] Text normalization WER matches benchmark scoring policy.
- [ ] Merging chunk transcriptions does not duplicate/drop words.

## 5. NLP-specific diagnostics

- [ ] Label2id/id2label map persisted; number of labels matches final layer.
- [ ] CE vs BCEWithLogits chosen for single/multi-label correctly.
- [ ] Attention mask used for pooled embedding (not padding contamination).
- [ ] Tokens/label shapes, target language and chat template correct.
- [ ] Train/val truncation rate measured; max input supports given architecture.
- [ ] Generation JSON is validated before scoring.
- [ ] No accidentally decoded prompt counted as answer.

## 6. Reproducibility & offline

- [ ] `seed` Python/NumPy/PyTorch and DataLoader workers if required.
- [ ] `torch.backends` settings documented (full determinism may affect performance).
- [ ] Save `requirements` versions, CUDA info, model and dataset revisions.
- [ ] Save best checkpoint, tokenizer/processor and label maps.
- [ ] **Cold reload test:** restart kernel and run inference without trainer state.
- [ ] **Offline test:** disable network; check model cache and artifacts, no API calls.
- [ ] `predictions` ids exactly match grader template; sorted in required order; no index shifts.
- [ ] Run one CPU/gpu fallback path if task requires portability.

## 7. Error analysis template (`error_analysis.csv`)

```
id,source_group,gold,pred,confidence,input_length,error_category,hypothesis,next_experiment
```

Lấy 20–50 lỗi định tính, kể cả lỗi confidence cao và low-resource groups. Chia theo nhãn, độ dài, domain, speaker, tiếng ồn/accents, tokenization, wrong formatting. Từ đó đưa ra **thí nghiệm cụ thể**; không chỉ liệt kê lỗi.

## 8. Artifact schema

**`metrics.json` minh họa cấu trúc** (dấu `null` = chưa thực nghiệm; đừng điền số tưởng tượng):

```json
{
  "run_id": "lab-aud03-whisper-baseline",
  "task": "asr",
  "model_id": "<exact-model-id>",
  "model_revision": "<sha-or-tag>",
  "data_revision": "<data-sha-or-version>",
  "split": "validation",
  "seed": 42,
  "primary_metric_name": "wer",
  "primary_metric_value": null,
  "cer": null,
  "rtf": null,
  "peak_vram_mb": null,
  "status": "NOT_RUN"
}
```


---

<!-- SOURCE_FILE: 06_AI_CODEGEN_PROMPTS.md -->

# Prompt Library — đưa Markdown này cho AI sinh code thực hành

> **Sử dụng:** đính kèm `README.md`, `00_SKILL_MATRIX.md`, tài liệu domain (`01_NLP_PRACTICE.md` hoặc `02_AUDIO_PRACTICE.md`), `04_TUNING_PLAYBOOK.md`, `05_EVALUATION_DEBUG.md`, config YAML tương ứng; sau đó copy prompt phù hợp bên dưới.
>
> Hãy thay `<...>` bằng thông tin thật; AI **không được tự bịa dataset, test result, API credential hoặc GPU**.

## PROMPT-00: System instruction — áp dụng cho mọi bài

```text
Bạn là ML engineer và huấn luyện viên IOAI (NLP/Audio). Dựa HOÀN TOÀN trên
các file Markdown và YAML được đính kèm, hãy tạo code Python/notebook
có thể chạy thực hành. Trước hết đọc skill IDs, acceptance criteria và
metric protocol. Không bỏ qua baseline; không tối ưu hyperparameter
trên test. Không dùng dataset/model không được cung cấp mà không ghi rõ
là gợi ý. Không giả vờ đã chạy hoặc tạo metric giả.

Output bắt buộc:
1. README.md: mục tiêu, dataset schema, cài đặt, GPU/CPU và command chạy.
2. solution.ipynb hoặc scripts `prepare.py, train.py, evaluate.py, infer.py`.
3. requirements.txt với versions tương thích và `config.yaml`.
4. Baseline đơn giản + phương án advanced theo task.
5. Evaluation primary metric, secondary metrics và error_analysis.csv.
6. Tuning sweep/ablation có cấu hình, lưu experiments.csv.
7. Logging seed, model/data revision, trainable params, time, peak GPU RAM.
8. Chỉ tạo metrics.json với `status=NOT_RUN` cho phần chưa thực thi.
9. Smoke tests cho 3 trường hợp bình thường và các edge cases.
10. Save/reload artifacts và thử offline inference nếu đề yêu cầu.

Với mỗi bước giải thích *tại sao cần* và *cách kiểm tra đúng*,
không chỉ đưa code. Nếu API đã thay đổi, ưu tiên docs của phiên bản
hiện được cài, không dùng argument deprecated mà không kiểm chứng.
```

## PROMPT-01: NLP text classification — `LAB-NLP-01`

```text
Hãy sinh notebook `LAB-NLP-01` từ `01_NLP_PRACTICE.md` và
`configs/nlp_classification.yaml`.
Dataset: <path / Hugging Face dataset>.
Columns: <text_col>, <label_col>.
Hardware: <GPU model / RAM / CPU>.

Bắt buộc: TF-IDF+LogisticRegression baseline, BERT/DistilBERT
classification full fine-tune, stratified/group split phù hợp,
proper tokenization/dynamic padding, macro-F1+confusion matrix,
error analysis và hyperparameter sweeps LR/length/epochs.
So sánh metric, elapsed, peak VRAM. Save best checkpoint + tokenizer;
standalone inference sau reload. Không tạo số đo giả.
```

## PROMPT-02: NLP embedding/retrieval — `LAB-NLP-02`

```text
Tạo training-free retrieval lab từ bộ queries/docs/relevance labels.
So sánh TF-IDF/BM25 với sentence embeddings, BERT CLS vs masked mean,
cosine similarity với L2 normalization; đánh giá Recall@1/5,
MRR@5 và query latency. Thêm tùy chọn hybrid + cross-encoder
re-ranking nếu hardware cho phép, *tách riêng* việc tuning dùng val.
Cần checkpoint pre-trained, data schema, ví dụ top-k retrieved docs
và lỗi retrieval có phân tích.
```

## PROMPT-03: Encoder–decoder — `LAB-NLP-04`

```text
Sinh bài học về T5 translation hoặc BART summarization
(dựa vào dataset được cho). Giải thích source/target tokenization,
`DataCollatorForSeq2Seq`, label masking, loss, generate(),
BEAM 1/4 và đánh giá BLEU/chrF hoặc ROUGE phù hợp.
Tạo baseline pre-trained, fine-tune trên train (budget được cung cấp),
validation checkpoint, save/reload, output sample/error analysis.
Không dùng test trong hyperparameter choice.
```

## PROMPT-04: LLM LoRA / SFT — `LAB-NLP-06`

```text
Sinh pipeline SFT một open-source LLM nhỏ phù hợp <VRAM GB>,
loại dataset <prompt-completion | messages>, dùng TRL+PEFT phiên bản
hiện hành; baseline zero-shot trước khi train. Kiểm tra chat template,
loss masking (assistant/completion only nếu phù hợp data), LoRA
r/alpha/dropout/target_modules, trainable params, lr, grad accumulation,
checkpointing. Đưa ablation low-rank vs high-rank (nếu có thời gian),
compute memory/time và reload adapter. Không dùng API/cloud trong
inference nếu exam offline.
```

## PROMPT-05: HuBERT classification — `LAB-AUD-02`

```text
Sinh notebook HuBERT audio classification theo
`02_AUDIO_PRACTICE.md` và `configs/audio_hubert_classification.yaml`.
Dataset schema audio/label/speaker_id (nếu có). Bắt buộc waveform
EDA, waveform resampling THẬT, duration plot, collator variable length,
feature extractor, masked pooling sau encoder, baseline frozen features
+ linear head và partial unfreeze HuBERT; macro-F1/confusion matrix;
augmentation ablation; trainable params/VRAM/time. Dùng split theo
speaker/recording nếu bài yêu cầu. Kiểm thử sample rate và empty audio.
```

## PROMPT-06: Whisper ASR fine-tune — `LAB-AUD-04`

```text
Sinh notebook ASR từ `configs/audio_whisper_asr.yaml`.
Hãy tạo baseline pre-trained Whisper inference + WER/CER,
WhisperProcessor + audio/text preprocessing, đúng resample/language/task,
custom collator đúng audio/label padding (-100), Seq2SeqTrainer hoặc
đoạn train loop tương thích API version, generation on validation,
checkpoint best theo WER. Tune learning rate, decoding beam, VAD,
normalization có kiểm soát; report WER/CER/RTF/VRAM, error cases
(noise, accent, long clips) và reload hoàn toàn offline.
Không chạy hoặc đưa ra thời gian/điểm giả.
```

## PROMPT-07: Qwen-Audio / Voxtral audio understanding

```text
Hãy xây notebook CHỈ INFERENCE để so sánh Qwen-Audio (chọn model
cụ thể có sẵn), Qwen2-Audio hoặc Voxtral phiên bản cụ thể.
Trước khi code, kiểm tra model card và API docs: model ID, processor,
chat template, supported task/audio length, local vs API,
GPU memory và license. Nếu checkpoint quá lớn, tạo adapter interface
hoặc chọn phiên bản/endpoint được phép, không giả chạy local.
Tập data gồm <audio_path, question, reference_answer>.
Tạo deterministic prompts, evaluate task accuracy/rubric,
format compliance, hallucination and latency. Có option
Whisper-transcript → text-LLM cascade baseline; không tiết lộ API key.
```

## PROMPT-08: IOAI 2026 — Find the Order (có grader)

```text
Truy cập tài liệu bài Find the Order tại:
https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/1_Find_the_Order
ĐỌC `statement.md`, `README.md`, `code/baseline-original` và
`code/grading-original` trước khi thiết kế phương án; chạy educational
baseline bên ngoài contest chỉ để phát triển.

Sinh notebook cải tiến với các mô-đun:
1) đọc shuffled WAV;
2) transcript (local ASR được phép dùng);
3) pairwise transition score dựa vào semantic/dialogue cues;
4) reconstruct order (greedy/beam/DP phù hợp constraints);
5) official/local grader adaptation và benchmark time/VRAM;
6) ablation: baseline, ASR-only, +semantic, +search.
Không tự bịa chi tiết input/output hoặc grader. Nếu thiếu file/data,
viết rõ chưa thể chạy và tạo interface có smoke tests.
```

## PROMPT-09: Training notebook generator — tạo giảng dạy

```text
Tạo một notebook giảng dạy 120 phút cho skill ID <NLP-.../AUD-...>.
Gồm: 10 phút mục tiêu; 20 phút ví dụ code tối thiểu; 45 phút guided lab;
30 phút tuning/ablation; 15 phút challenge và đáp án.
Cho học viên chỗ `# TODO` có unit test; tạo instructor solution
riêng; rubric chấm 100 điểm; cùng một môi trường Python.
Không để lộ lời giải trong student notebook.
```

## PROMPT-10: Contest-mode stress test

```text
Biến solution hiện có thành bản dự thi offline có giới hạn:
GPU <= <VRAM> GB, runtime <= <time> giây, disk <= <disk> GB,
không Internet, input/output theo statement chính thức.
Đọc grader, preflight artifacts đã cache, đo wall-clock và peak VRAM,
có fallback baseline, không pip install tại runtime,
kiểm tra trên 5 edge cases (empty/long/noise/Unicode/batch size 1),
kiểm tra deterministic submission format và cold restart.
Báo cáo những điều chưa kiểm chứng, không điền số liệu tưởng tượng.
```

## PROMPT-11: Tuning assistant theo evidence

```text
Từ `experiments.csv`, `error_analysis.csv`, train/val curves và
`04_TUNING_PLAYBOOK.md`, hãy phân tích nguyên nhân lỗi.
Đề xuất tối đa 5 thí nghiệm tiếp theo, theo thứ tự expected gain /
cost, ghi hypothesis, thay đổi duy nhất, metric, runtime estimate
là ước tính (nếu thực sự có dữ liệu), stopping criterion.
Cấm sử dụng test set để chọn config; cấm khẳng định improvement
nếu chưa chạy; nhớ check split leakage/sample rate/processor trước.
```

## Tiêu chuẩn output khi AI sinh xong

1. Chạy `import` + load 3 samples thành công, in shape/dtype/device.
2. Baseline đúng metric và label mapping.
3. Train chạy ít nhất 1 step/epoch (nếu có đủ dữ liệu), save checkpoint.
4. Reload từ ổ đĩa, inference không còn phụ thuộc biến trong notebook.
5. Generated file `predictions.csv` phù hợp template grader.
6. Làm rõ những gì là ví dụ, những gì đã thực sự được đo.


---

<!-- SOURCE_FILE: 07_REPOS_RESOURCES.md -->

# Repos & Docs tham khảo — ưu tiên code và notebook chính thức

> Kiểm tra sự tồn tại của các repo GitHub chính dưới đây ngày 2026-10-23. Một số ví dụ cũ trên Internet dùng API đã đổi; **luôn đọc `README`, model card, requirements và ví dụ của phiên bản đang cài**. Danh sách không đồng nghĩa mọi repo đều có code training cho mọi model.

## A. Bài thi IOAI — BẮT ĐẦU Ở ĐÂY

| Nguồn | Nội dung nên học | URL |
|---|---|---|
| IOAI 2026 official contest | 6 nhiệm vụ, notebook/baseline/grader | https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest |
| **Find the Order** | Audio+NLP: transcribe shuffled turns, reconstruct dialogue; code baseline/eval | https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/1_Find_the_Order |
| Find the Order educational baseline notebook | Mở trực tiếp trong Colab | https://colab.research.google.com/github/IOAI-official/IOAI-2026/blob/main/Individual-Contest/1_Find_the_Order/code/baseline/solution.ipynb |
| Find the Order educational grader | Chấm nghiệm outside contest | https://colab.research.google.com/github/IOAI-official/IOAI-2026/blob/main/Individual-Contest/1_Find_the_Order/code/grading/evaluate.ipynb |
| Find the Order HF dataset | Public/private splits, đọc license/card | https://huggingface.co/datasets/IOAI-official/ioai-2026-find-the-order |
| **Potato** | Semantic word guessing / ranking / embeddings | https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/3_Potato |
| **Ghost of the Machine** | Machine-edited passage detection | https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/5_Ghost_of_Machine |
| **Robot Chasing** | Natural language → behavior | https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest/2_Robot_Chasing |
| IOAI 2025 `Concepts` | LLM/API, official solution + simulated judge API | https://github.com/IOAI-official/IOAI-2025/tree/main/Individual-Contest/Concepts |
| IOAI 2025 all tasks | Notebook practice và đáp án tham khảo | https://github.com/IOAI-official/IOAI-2025 |
| IOAI 2024 tasks | Đề thi trước đây; không nhất thiết về NLP/audio | https://github.com/IOAI-official/IOAI-2024/tree/main/On-Site-Round |

**Quan trọng:** trong IOAI 2026, `baseline-original` và `grading-original` là mã thi gốc, còn `baseline`/`grading` là các bản educational được cung cấp để chạy ngoài contest. Code educational có thể tải data từ HF, nhưng bài thi gốc không cho Internet. Dataset của từng task thường nằm trên HF, không nằm trong Git repo.

## B. NLP — Library và code chính thức

| Repo / docs | Nên tham khảo cho | URL |
|---|---|---|
| Hugging Face Transformers | BERT/encoder, seq2seq, LM, Trainer/pipelines | https://github.com/huggingface/transformers |
| Text classification task guide | DistilBERT fine-tuning, Trainer, metrics, collator | https://huggingface.co/docs/transformers/main/tasks/sequence_classification |
| Translation task guide | T5 seq2seq, translation dataset/collator | https://huggingface.co/docs/transformers/tasks/translation |
| HF Datasets | Load/map/split/Audio features | https://github.com/huggingface/datasets |
| HF Evaluate | Task evaluation | https://github.com/huggingface/evaluate |
| HF Notebooks | Notebook ví dụ nhiều loại task | https://github.com/huggingface/notebooks |
| HF PEFT | LoRA, prompt/adapters | https://github.com/huggingface/peft |
| HF TRL | SFT Trainer, PEFT integration | https://github.com/huggingface/trl |
| TRL SFT docs | Dataset formats, completion-only loss, LoRA | https://huggingface.co/docs/trl/sft_trainer |
| HF SetFit | Few-shot text classification | https://github.com/huggingface/setfit |
| Sentence Transformers (official repo) | Sentence embeddings, semantic similarity/retrieval | https://github.com/UKPLab/sentence-transformers |

## C. Audio — Preprocessing, HuBERT, Whisper, Qwen, Voxtral

| Repo / docs | Kỹ năng / ví dụ | URL |
|---|---|---|
| **HF Audio Transformers Course** | Toàn bộ pipeline audio, bài tập classification, ASR | https://github.com/huggingface/audio-transformers-course |
| HF Audio Course — HuBERT assignment | HuBERT music classification, accuracy tuning | https://huggingface.co/learn/audio-course/en/chapter4/hands_on |
| **HF Audio Course — fine-tune Whisper** | Common Voice ASR, WhisperProcessor, Seq2SeqTrainer | https://huggingface.co/learn/audio-course/chapter5/fine-tuning |
| Audio classification guide | Wav2Vec2 MInDS-14 classification (chuyển kiến thức sang HuBERT) | https://huggingface.co/docs/transformers/main/tasks/audio_classification |
| ASR task guide | Audio resampling, processor, CTC/ASR workflow | https://huggingface.co/docs/transformers/main/tasks/asr |
| OpenAI Whisper | ASR model & basic decode | https://github.com/openai/whisper |
| SYSTRAN faster-whisper | CTranslate2 inference; batching, quantization settings | https://github.com/SYSTRAN/faster-whisper |
| PyTorch Audio | Audio processing/training primitives; check API compatibility | https://github.com/pytorch/audio |
| SpeechBrain | Speech classification/ASR recipes, hyperparam YAML | https://github.com/speechbrain/speechbrain |
| NVIDIA NeMo | ASR / speech recipes, larger-scale resources | https://github.com/NVIDIA/NeMo |
| silero-vad | Voice activity detection | https://github.com/snakers4/silero-vad |
| WhisperX | ASR timestamps, alignment, diarization workflows (cần kiểm tra license/capabilities) | https://github.com/m-bain/whisperX |
| **Qwen-Audio (original)** | Audio QA/caption/audio understanding | https://github.com/QwenLM/Qwen-Audio |
| **Qwen2-Audio** | Thế hệ khác; dùng đúng processor/template | https://github.com/QwenLM/Qwen2-Audio |
| Qwen2.5-Omni | Nâng cao: omni-modal, không đồng nhất Qwen-Audio gốc | https://github.com/QwenLM/Qwen2.5-Omni |
| **Mistral Voxtral docs** | Dòng Voxtral, chọn model/API theo phiên bản, inference audio | https://docs.mistral.ai/studio/audio/overview |
| Mistral cookbook | Ví dụ Mistral API/SDK, tìm notebook tương thích model | https://github.com/mistralai/cookbook |
| Mistral inference | Hạ tầng inference; không mặc định bao gồm mọi Voxtral model | https://github.com/mistralai/mistral-inference |

**Cẩn trọng với Voxtral:** trang tài liệu Mistral có thể đã nâng lên Voxtral 3; mã gọi Voxtral đời trước có thể không tương thích. Không chốt model ID/endpoint trước khi xác nhận docs phù hợp tài khoản/môi trường. **Cẩn trọng với Qwen:** bản gốc Qwen-Audio không thể thay bằng Qwen2-Audio hay Qwen2.5-Omni chỉ bằng cách đổi model name.

## D. Link syllabus chính thức

- PDF 2026: https://ioai-official.org/wp-content/uploads/2025/10/Syllabus.pdf
- Official resource portal: https://ioai-official.org/resources/
- Scope trong PDF: NLP text classification (Practice), pre-trained encoder BERT (Both), language modeling (Both), encoder–decoder (Practice), pre-trained language models local/API (Practice); audio HuBERT, Qwen-Audio, Whisper, Voxtral (Practice).

## E. Ưu tiên đọc repo theo thời gian

1. Ngày 1: `huggingface/transformers` text classification guide; làm baseline trong lab NLP.
2. Ngày 2: HF audio course HuBERT + audio classification.
3. Ngày 3: HF audio course Whisper FT + faster-whisper inference.
4. Ngày 4: `IOAI-2026/1_Find_the_Order` baseline **và** grader; xem cả scoring rule.
5. Ngày 5: `Qwen-Audio`, `Qwen2-Audio`, Mistral Voxtral docs; so task và hardware compatibility.
6. Sau đó: `peft`, `trl`, SpeechBrain và các bài text khác (Potato, Ghost, Concepts).

## F. Cách đọc một repo hiệu quả

1. Kiểm tra README/requirements, Python/CUDA/model version, license.
2. Tìm **một minimal inference example** chạy được.
3. Xem dataset schema + preprocessor + forward signature + output units.
4. Xem trainer, collator, metric, evaluation script (nếu có).
5. Chạy 5–10 samples smoke test, sau đó mở rộng; kiểm tra Memory/Runtime.
6. Ghi vào note: `repo_url`, `commit_sha`, `example_path`, `model_id`, `what_learned`, `gotchas`.
