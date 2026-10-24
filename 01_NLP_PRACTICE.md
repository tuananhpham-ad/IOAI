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
