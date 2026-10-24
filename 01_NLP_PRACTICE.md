# NLP — Học và thực hành theo dạng bài

> Mỗi module cần có notebook/script, dataset schema, metrics, baseline, tuning và phân tích lỗi. Phân biệt minh họa từ dữ liệu thực nghiệm.

> Bổ sung có nguồn ngày **2026-10-24**: NLP-BS-01 → NLP-BS-06 đi sâu vào preprocessing, augmentation, model nhỏ, ensemble và nén model. Model card/paper là nguồn cho thông số; các recipe và khoảng tuning là đề xuất thực hành, chưa phải kết quả benchmark trong repo. Practice CV: [03_CV_PRACTICE.md](03_CV_PRACTICE.md).

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

## NLP-BS-01 — Preprocessing theo task, không làm sạch máy móc (P0)

**Thứ tự:** kiểm tra schema/nhãn → tìm nhóm trùng/nguồn chung → chia split → fit biến đổi trên train → tokenize → kiểm tra nội dung bị cắt. Mọi bản augment/chunk của một mẫu phải ở cùng split với mẫu gốc.

| Tình huống | Cách thử | Lỗi cần tránh / kiểm chứng |
|---|---|---|
| Unicode, dấu tiếng Việt | So raw với Unicode NFC; chuẩn hóa line ending; giữ bản raw để đối chiếu | NFKC có thể đổi ký hiệu; bỏ dấu thường mất nghĩa; không tự lowercase model cased |
| HTML, URL, email, user mention | Parse HTML khi markup không phải tín hiệu; thử thay URL/mention bằng marker nhất quán | Marker mới có thể bị tokenizer chia nhỏ; không xóa URL nếu domain là tín hiệu của nhãn |
| Emoji, dấu câu, chữ lặp | Giữ nguyên baseline; chỉ giảm lặp/chuẩn hóa nếu val chứng minh có ích | Sentiment, sarcasm và phát hiện văn bản máy có thể cần chính những dấu hiệu này |
| Tiếng Việt với PhoBERT | Word segmentation trước BPE theo model card; ví dụ `sinh_viên`, không áp dụng pipeline này cho mọi model | Raw text và segmented text không thay thế nhau; kiểm tra giới hạn token của checkpoint |
| TF-IDF trên text noisy | Word n-gram `(1,2)` + char n-gram `(3,5)`; thử `min_df`, `max_df`, `sublinear_tf` | Fit vocabulary/IDF trong từng train fold; TF-IDF không có số tham số cố định như Transformer |
| Văn bản dài | So head-only, head+tail, sentence chunks; khởi đầu 128/256/512 token, overlap 10–20% khi phù hợp | Gom dự đoán ở document level; overlap dễ đếm lặp evidence; document dài không được tự chiếm nhiều trọng số |
| NER / extractive QA | Giữ offset mapping, align nhãn với subword; các token bỏ qua dùng `-100` | Normalization thay offset: phải cập nhật span; không chấm token bị pad |
| Retrieval | Tách query/document; chunk có `doc_id`, `chunk_id`; đúng prefix/pooling của checkpoint | E5 cần `query: ` / `passage: ` kể cả non-English; không gắn prefix E5 vào tất cả model |
| Dữ liệu lệch domain | Group split theo tác giả, sách, user hoặc thời gian; báo cáo theo từng nhóm | Random split câu cùng sách có thể đánh giá quá lạc quan |

Các yêu cầu riêng của tokenizer/prefix xem [PhoBERT](https://huggingface.co/vinai/phobert-base-v2) và [multilingual E5](https://huggingface.co/intfloat/multilingual-e5-small). Với E5, dùng masked mean pooling và normalize theo ví dụ chính thức; không lấy CLS tùy tiện.

**Lab `LAB-NLP-07`:** giữ một checkpoint và split; chạy raw → NFC → normalization theo domain → chunking. Ghi macro-F1/MRR theo task, truncation rate, số token và latency. Xuất 20 cặp trước/sau để kiểm tra nhãn và span. Không mặc định pipeline dài nhất là tốt nhất.

## NLP-BS-02 — Augmentation: ưu tiên bảo toàn nhãn (P0/P1)

Augment chỉ trên train. Với câu “không tốt”, xóa “không” làm đổi nhãn; với bài phát hiện đoạn máy sửa, paraphrase có thể xóa chính dấu vết cần học. Các mức dưới đây là **điểm bắt đầu để ablate**, không phải thông số tối ưu từ paper.

| Cách làm | Khi nên thử | Cấu hình khởi đầu / điều kiện |
|---|---|---|
| Character noise: typo, thiếu dấu, OCR confusion | Input thực tế có lỗi gõ/OCR | Tác động 1–3% ký tự trên 10–20% mẫu; giữ tên, số, phủ định nếu mang nhãn |
| EDA: synonym replacement, insertion, swap, deletion | Classification ít nhãn, đặc biệt topic | Một phép/lần, khoảng 5% từ; kiểm tra từ đồng nghĩa đúng ngữ cảnh và ngôn ngữ |
| Contextual substitution bằng masked LM | Cần biến thể tự nhiên hơn synonym list | Thay ít token, khóa entity/number/negation; teacher phải phù hợp ngôn ngữ |
| Back-translation | Classification hoặc tăng đa dạng diễn đạt | Một bản/mẫu trước; lọc đổi entity, số, phủ định và câu trùng; tính cả thời gian sinh |
| Paraphrase có ràng buộc bằng LLM | Có compute chuẩn bị dữ liệu và task cho phép | Yêu cầu giữ facts/nhãn; audit thủ công theo lớp; teacher confidence không thay thế kiểm tra |
| Template / entity replacement | Intent, slot filling, dữ liệu có cấu trúc | Thay entity cùng loại và cập nhật BIO/span; không đổi entity nếu nó quyết định label |
| Sentence crop / reorder | Tài liệu dài với nhãn thật sự bất biến theo đoạn | Không dùng reorder cho temporal reasoning, coherence, QA hoặc hội thoại cần thứ tự |
| Pair sampling / hard negatives | Similarity, retrieval, few-shot classification | Sinh pairs trong train; lọc false negatives; cùng nhãn topic chưa chắc cùng ý nghĩa |
| Pseudo-label + consistency (UDA) | Có dữ liệu unlabeled được phép dùng | Chọn ngưỡng qua val, kiểm soát số mẫu/lớp; teacher weak view → student augmented view |
| Mixup trên embeddings | Có custom training loop, muốn regularize classifier | Trộn hidden vectors và soft labels; không trộn số token ID; chỉ là thí nghiệm P1 |
| R-Drop | Overfit; muốn regularization không sửa câu | Hai forward với dropout, supervised loss + symmetric KL; tốn thêm train compute, inference vẫn một model |

Nguồn phương pháp: [EDA](https://aclanthology.org/D19-1670/), [UDA với back-translation/consistency](https://arxiv.org/abs/1904.12848), [R-Drop](https://arxiv.org/abs/2106.14448), [SetFit cho few-shot sentence classification](https://huggingface.co/docs/setfit/index). R-Drop là regularization, SetFit là quy trình fine-tune; không gọi chúng là phép sửa raw text.

**Protocol lọc:** lưu `parent_id, transform, seed, teacher_revision`; loại bản trùng; kiểm tra facts và nhãn trên mẫu đại diện; cố định số optimizer steps khi so có/không augment. Nếu chỉ tăng số epoch/tokens thì chưa tách được lợi ích của augmentation.

**Lab `LAB-NLP-08`:** baseline → noise nhẹ → một phép EDA hoặc paraphrase → R-Drop (nếu đủ thời gian). Mỗi run chỉ thêm một thành phần; đo clean val và một bộ noisy val tạo riêng sau split. Chọn theo metric mục tiêu, không theo số mẫu sinh được.

## NLP-BS-03 — Model nhỏ: baseline cực nhẹ và ứng viên SOTA theo ngân sách (P0)

### Cách đọc bảng

Tra cứu nguồn chính thức ngày **2026-10-24**. Đây là shortlist có thể thực hành, không phải chứng nhận model đứng đầu mọi leaderboard. **SOTA phải gắn với task, ngôn ngữ, benchmark và giới hạn tài nguyên.** M = triệu, B = tỷ tham số; số gần đúng thay đổi theo head và cách tính tied embeddings. Các model trong cùng bảng chưa được benchmark trực tiếp trong repo.

### Encoder / embeddings: chọn trước LLM nếu task cho phép

| Checkpoint / nguồn | Params ≈ | Vai trò | Khi chọn / giới hạn |
|---|---:|---|---|
| TF-IDF word+char + LogisticRegression | Phụ thuộc vocabulary × classes | Baseline CPU | Rất đáng thử với topic, spam, dấu vết từ vựng; đo file vectorizer và classifier |
| [Google BERT L-2 H-128](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) | 4.4M | Encoder cực nhỏ, baseline cũ | English uncased; fine-tune head/task; không phải sentence embedding sẵn dùng |
| [Ettin encoder 17M](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | 17M | Encoder nhỏ thế hệ mới | Thử classification/token tasks tiếng Anh; cần fine-tune; điểm của Ettin Base/Large không phải điểm của bản 17M |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22.7M | Sentence embedding baseline | English, vector 384 chiều; mặc định truncate sau 256 wordpieces; hợp frozen embeddings + linear/SetFit |
| [Granite embedding small English r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) | 47.7M | Retrieval English hiện đại | Vector 384, context tới 8192; ứng viên khi cần văn bản dài hơn MiniLM; đọc đúng recipe encode |
| [Granite embedding 97M multilingual r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) | 97.4M | Retrieval đa ngôn ngữ | Vector 384, context tới 32768; có tiếng Việt trong nhóm ngôn ngữ được tăng cường |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Khoảng 118M | Baseline retrieval đa ngôn ngữ | Vector 384, max 512 token; dùng đúng prefix; chữ “small” không có nghĩa <30M |
| [PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2) | Khoảng 135M | Encoder chuyên tiếng Việt | Word segmentation; phù hợp làm mốc so classification/NER tiếng Việt, không phải lựa chọn nhỏ nhất |
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 149M | Encoder English context dài | Tới 8192 token; mốc chất lượng khi encoder nhỏ thiếu khả năng; raw MLM không tự thành retriever tốt |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 600M | Embedding đa ngôn ngữ lớn hơn | Vector tới 1024, hỗ trợ 32–1024 chiều; chỉ nâng lên nếu metric bù được compute; giảm chiều vector không giảm trọng số model |

**Bằng chứng có phạm vi:** IBM báo cáo Granite 97M multilingual r2 đạt 60.3 trên Multilingual MTEB Retrieval (18 tasks), so với 50.9 của multilingual-e5-small trong bảng của họ. Đây là số do tác giả công bố theo protocol của họ, không phải kết quả riêng cho tiếng Việt hoặc bảo đảm thắng trên dữ liệu contest. Xem [model card và bảng evaluation](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2).

### Generation: từ rất nhỏ lên dưới 1B

| Checkpoint / nguồn | Params ≈ | Mục tiêu thực hành | Giới hạn cần kiểm tra |
|---|---:|---|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135M | Baseline sinh text cực nhỏ; extraction/format đơn giản | Thiên English; reasoning và kiến thức hạn chế; có thể thử 360M cùng dòng nếu thiếu chất lượng |
| [Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it) | 270M | Task hẹp, fine-tune/distill, output ngắn | Text-only ở kích thước này; cần chấp nhận điều khoản tải weights; không suy khả năng vision từ tên Gemma 3 |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 600M | Đa ngôn ngữ, structured output, local SFT | So thinking/non-thinking; giới hạn token sinh; đọc chat template và decode recommendations |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | LM 800M; còn vision encoder | Ứng viên mới cho text và image-text | Card ghi 0.8B ở phần language model, không lấy làm tổng VLM; processor/backend khác Qwen3; ưu tiên prototyping/task-specific FT |

Không cần nhảy lên 3–7B trước khi thử baseline đúng task. Với classification, so **TF-IDF → encoder frozen → encoder fine-tune**; với retrieval tiếng Việt, thử **BM25 → Granite 97M/E5-small → hybrid**; với generation, thử checkpoint nhỏ nhất đáp ứng format và accuracy. Đây là thứ tự thử nghiệm đề xuất, không phải xếp hạng chất lượng đã đo.

### “Nhỏ” phải đo đủ bốn thứ

1. **Weights:** ước tính lý tưởng `params × bits / 8`; 100M ở FP16 ≈ 200 MB, INT8 ≈ 100 MB. Metadata, scales và tensor không quantized làm artifact lớn hơn.
2. **RAM/VRAM lúc chạy:** activations, tokenizer, KV cache, batch, sequence length; LoRA ít trainable params nhưng vẫn cần base weights.
3. **Latency:** p50/p95 batch=1 và throughput batch thực tế; có warm-up, đồng bộ GPU khi đo; báo CPU/GPU/backend/dtype.
4. **Chất lượng:** cùng split/scorer, theo ngôn ngữ/độ dài/lớp; chọn model nhỏ nhất đạt ngưỡng metric và runtime đã đặt.

## NLP-BS-04 — Ensemble có kiểm soát chi phí (P1)

| Cách | Recipe | Chi phí / điều kiện |
|---|---|---|
| Soft voting | `p = Σ w_i p_i`, `w_i >= 0`, `Σ w_i = 1`; bắt đầu đều trọng số | Căn đúng sample IDs và thứ tự labels; classification single-label dùng softmax, multi-label dùng sigmoid |
| Lexical + neural | TF-IDF LogisticRegression + encoder, thử weight `[0, .25, .5, .75, 1]` trên val | Có thể bổ sung lỗi khác nhau; nếu dùng SVM raw margin phải calibrate trước khi trộn probabilities |
| Seed / fold ensemble | 2 seeds hoặc 3 folds trước; chỉ giữ thành viên có lợi | Tổng thời gian inference tăng theo số model; disagreement cao chưa đủ chứng minh ensemble tốt |
| Hybrid retrieval | BM25 + dense; RRF: `score(d)=Σ 1/(k+rank_i(d))`, rank bắt đầu từ 1, thử `k=60` | Tài liệu không có trong list đóng góp 0; RRF tránh cộng raw BM25 với cosine khác scale |
| Stacking | Fit meta-model trên out-of-fold predictions; mọi transform fit bên trong fold | Đánh giá cuối trên holdout riêng/nested CV; không fit rồi báo score ngay trên cùng OOF dùng train meta-model |
| Generation voting | Sinh ít ứng viên rồi chọn bằng format checks/rubric/reranker độc lập | Không average token logits giữa tokenizer/vocabulary khác nhau; nhiều đáp án sai giống nhau vẫn thắng vote |
| EMA / SWA / model soup | EMA một trajectory; SWA các checkpoint; soup các lần FT cùng pretrained init | Deploy một model; cùng architecture/head/tokenizer; weight averaging không tương đương probability averaging |

Với binary/multi-label, tune threshold sau khi tạo ensemble. Nếu vừa fit calibration, vừa chọn weights và threshold trên val nhỏ, dễ overfit; dùng OOF hoặc tách calibration/selection khi dữ liệu cho phép. Giữ test độc lập.

**Model soup:** chọn greedy soup trên val, chỉ thêm checkpoint khi có ích; không average TF-IDF với BERT hay các model khởi tạo độc lập không được căn chỉnh. Phương pháp và điều kiện gốc: [Model soups](https://arxiv.org/abs/2203.05482).

**Lab `LAB-NLP-09`:** TF-IDF, encoder nhỏ, blend hai model; lưu probabilities theo `sample_id`; so từng model và blend trên cùng holdout. Nộp bảng `metric, total_params, artifact_mb, peak_memory, latency_p95`; nếu blend vượt budget, dùng làm teacher cho NLP-BS-05.

## NLP-BS-05 — Distillation và tối ưu triển khai (P1)

- **Classification distillation:** teacher/ensemble tạo soft targets trên train; student nhỏ học `L = (1-α) CE(y, p_s) + α T² KL(p_t(T) || p_s(T))`. Thử `α=[0.25,0.5,0.75]`, `T=[1,2,4]`; teacher ở eval/no-grad. Với multi-label dùng objective phù hợp Bernoulli, không ép softmax giữa các nhãn độc lập. Nguồn: [Knowledge distillation](https://arxiv.org/abs/1503.02531).
- **Generation distillation:** teacher tạo completion rồi SFT student trên cặp prompt/completion đã lọc. Cách này dùng được khi tokenizer khác; logit distillation trực tiếp cần căn chỉnh vocabulary.
- **INT8 / INT4 inference:** so FP32/FP16/BF16 với backend có kernel hỗ trợ đúng kiến trúc. INT4 không mặc định nhanh hơn, nhất là encoder nhỏ/CPU. Calibration cho PTQ chỉ dùng train/calibration split.
- **Giảm context trước tăng quantization:** chọn length đủ evidence, bucket theo length, dynamic padding, cache frozen embeddings. Không cache một lần rồi gọi đó là online augmentation nếu muốn biến thể thay đổi mỗi epoch.
- **Pruning:** unstructured sparsity không tự giảm latency/file; structured pruning cần backend/shape phù hợp và fine-tune lại. Ưu tiên student nhỏ sẵn có nếu thời gian ít.
- **Retrieval index:** FP16/INT8 vector hoặc giảm chiều với checkpoint hỗ trợ Matryoshka; đo Recall@k sau nén, renormalize khi cần. Đây là nén index, không phải nén model encoder.

**Exit:** reload artifact trong process mới, đủ tokenizer/config/label map, test offline trên 10 mẫu và input dài/rỗng; ghi cả base checkpoint nếu adapter chưa merge.

## NLP-BS-06 — Lộ trình thực hành theo ngân sách

| Budget / mục tiêu | Thứ tự ưu tiên |
|---|---|
| CPU, cần rất nhỏ | TF-IDF → BERT 4.4M/Ettin 17M cho classification; MiniLM 22.7M cho English embeddings |
| Ít nhãn | Frozen embeddings + linear → SetFit/pair sampling → augment có audit |
| Tiếng Việt | TF-IDF char+word → encoder/embedding có hỗ trợ Việt; PhoBERT cần segmentation; so Granite 97M và E5 cho retrieval |
| Có GPU, vẫn ưu tiên gọn | Partial/full FT model nhỏ → augmentation nhẹ → EMA hoặc blend 2 model → distill nếu blend quá chậm |
| Bài inference chỉ vài phút | Cache hợp lệ, batch, length/prompt/decode, hybrid retrieval; tránh bắt đầu pipeline sinh dữ liệu/FT nặng |

**Bảng ablation bắt buộc:** baseline → preprocess tốt nhất → + một augmentation → + một ensemble → student/quantized. Mỗi dòng ghi metric, seed, model revision, thời gian train/infer, peak RAM/VRAM và dung lượng artifact. Cho phép kết luận “không cải thiện”; không điền số benchmark chưa chạy.

## NLP-10 — Preprocessing văn bản (P0)

> Nguyên tắc: preprocessing phải **khớp với cách checkpoint được pretrain** (đọc model card/tokenizer). Mọi bước "fit" (vocab, TF-IDF, thống kê) chỉ fit trên train.

| Bước | Khi nào dùng | Cạm bẫy |
|---|---|---|
| Unicode normalize `NFC` | Luôn làm với tiếng Việt (dấu tổ hợp vs dựng sẵn cho ra token khác nhau) | `NFKC` có thể đổi ký tự full-width/ký hiệu — kiểm tra trước |
| Word segmentation (`underthesea`, `pyvi`, VnCoreNLP RDRSegmenter) | **Bắt buộc với PhoBERT** (pretrain trên text đã tách từ `học_sinh`) | Không tách từ khi dùng XLM-R/mmBERT/ViSoBERT |
| Lowercase | Chỉ với model *uncased* hoặc TF-IDF | Model cased mất thông tin tên riêng |
| Thay URL/email/số/@user bằng placeholder | Text mạng xã hội, noisy | Xóa mất tín hiệu nếu nhãn phụ thuộc (spam, phishing) |
| Emoji → giữ hoặc `emoji.demojize` | Sentiment | Tokenizer cũ có thể map emoji thành `[UNK]` — kiểm tra tỷ lệ UNK |
| Dedupe exact + near-dup (MinHash/`datasketch`, hoặc cosine embedding > 0.95) | Trước khi split | Near-dup giữa train/val → val ảo cao |
| Head+tail truncation (vd. 128 đầu + 382 cuối) | Văn bản dài, kết luận ở cuối (review, bài báo) | Thử ablation với head-only và sliding window |
| Chuẩn hóa teencode/viết tắt (`ko`→`không`) | Dữ liệu chat tiếng Việt | Dictionary quá rộng làm sai nghĩa; ablation |
| Loại câu rỗng/quá ngắn, sửa label noise (cleanlab, loss cao bất thường) | Dataset crowd-sourced | Không xóa mẫu khó chỉ vì loss cao — kiểm tra tay |

```python
import unicodedata, re
def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"https?://\S+", " <url> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
# PhoBERT: from underthesea import word_tokenize; word_tokenize(t, format="text")
```

## NLP-11 — Data augmentation cho NLP (P1)

> Augment **chỉ trên train**, giữ nguyên nhãn (label-preserving), luôn so sánh với baseline không augment trên cùng val. Với model pretrained mạnh, augment thường chỉ có lợi khi data ít (< vài nghìn mẫu) hoặc lệch lớp.

| Kỹ thuật | Ý tưởng | Hợp với | Rủi ro |
|---|---|---|---|
| EDA (synonym replace, random insert/swap/delete, p≈0.1) | Nhiễu cấp từ, rẻ | Data rất ít, TF-IDF/LSTM | Đổi nghĩa với negation ("không tốt") |
| Back-translation (vi→en→vi) | Paraphrase tự nhiên | Classification, NLI | Chậm; mất tên riêng; dùng `facebook/nllb-200-distilled-600M` hoặc `Helsinki-NLP/opus-mt-*` |
| Contextual substitution (MLM điền `[MASK]`, `nlpaug`) | Thay từ theo ngữ cảnh | Sentiment, topic | MLM có thể điền từ trái nghĩa |
| LLM paraphrase / generate mẫu mới | Few-shot → nhiều mẫu đa dạng | Lớp hiếm, intent detection | Phải lọc lại (classifier/LLM-judge); phân phối lệch so với test |
| Token/span masking, word dropout | Regularize | Fine-tune encoder | Mask quá nhiều phá câu |
| Noise tiếng Việt: bỏ dấu, lỗi telex, viết tắt | Robust với text thực tế | Social media, chat | Không áp dụng nếu test sạch |
| Embedding Mixup / Manifold Mixup | Nội suy hidden state + nhãn | Classification | Không hợp cho token-level (NER) |
| R-Drop (2 lần forward, KL giữa 2 dropout) | Consistency regularization | Fine-tune mọi encoder | x2 compute |
| Adversarial training FGM/AWP trên embedding | Nhiễu gradient | Thi đấu (Kaggle classic) | Tăng thời gian ~1.5–2x |
| Pseudo-labeling / self-training | Gán nhãn unlabeled bằng teacher, giữ confidence cao | Có nhiều text không nhãn | Khuếch đại lỗi teacher; không dùng test vào train nếu luật cấm |
| NER: mention replacement, label-wise token replacement | Thay entity cùng loại | Token classification | Phải giữ alignment BIO |

```python
# Back-translation nhanh với NLLB (chỉ chạy trên train)
from transformers import pipeline
fwd = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="vie_Latn", tgt_lang="eng_Latn")
bwd = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="vie_Latn")
def back_translate(texts):
    en = [o["translation_text"] for o in fwd(texts, max_length=256)]
    return [o["translation_text"] for o in bwd(en, max_length=256)]
```

## NLP-12 — Ensemble cho NLP (P1)

| Cách | Chi phí | Ghi chú |
|---|---|---|
| Multi-seed cùng model, trung bình **xác suất** (softmax) | Train x N | Rẻ nhất, thường +0.5–1 điểm F1 |
| K-fold (5 fold) + trung bình dự đoán test | Train x K | Có OOF prediction để tune trọng số/threshold |
| Đa dạng backbone (ModernBERT + DeBERTa + PhoBERT…) | Train x N | Đa dạng càng cao, gain càng lớn; chọn qua correlation OOF |
| Blend với TF-IDF+LR / char n-gram | Rất rẻ | Bổ sung tín hiệu lexical, hữu ích khi text noisy |
| Weighted average, tối ưu trọng số trên OOF (`scipy.optimize`, hill climbing) | Rẻ | Không tune trên test; ít model thì giữ trọng số đơn giản |
| Rank averaging | Rẻ | Khi metric là AUC và các model có calibration khác nhau |
| Stacking (LR/LightGBM trên OOF logits) | Trung bình | Dễ overfit nếu ít dữ liệu |
| Checkpoint averaging / SWA / EMA weights | ~0 | "Ensemble trong 1 model", không tăng latency |
| Model soup (trung bình weights nhiều fine-tune cùng init) | ~0 | Chỉ hợp khi cùng pretrained init và hyperparams gần |
| LLM: self-consistency (sample k lần, majority vote), nhiều prompt | k lần inference | Dùng cho reasoning/extraction có đáp án rời rạc |
| Threshold tuning sau ensemble | ~0 | Binary/multi-label: tune theo F1 trên OOF/val |

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
# oof: list các mảng [N, C] xác suất OOF của từng model; y: nhãn thật
def neg_f1(w):
    w = np.abs(w) / np.abs(w).sum()
    p = sum(wi * o for wi, o in zip(w, oof))
    return -f1_score(y, p.argmax(1), average="macro")
w = minimize(neg_f1, np.ones(len(oof)) / len(oof), method="Nelder-Mead").x
```

Checklist: mọi model trong ensemble dùng **cùng split/fold**; báo cáo điểm từng model + ensemble; đo latency tổng nếu đề có giới hạn thời gian.

## NLP-13 — Model nhỏ gọn hiện tại (2025–2026) cho NLP

> Kiểm tra model card, license, revision trước khi dùng; số tham số là xấp xỉ. Ưu tiên model nhỏ nhất đạt baseline, rồi mới scale lên.

### Encoder (classification, NER, token tasks)

| Model (HF ID) | Params | Ghi chú |
|---|---|---|
| `jhu-clsp/ettin-encoder-17m` / `-32m` / `-68m` / `-150m` | 17M–150M | Bộ Ettin (2025): encoder tiếng Anh, vượt ModernBERT cùng cỡ trên GLUE/MTEB; bản 17M/32M rất hợp khi GPU yếu |
| `answerdotai/ModernBERT-base` | 149M | Context 8192, FlashAttention, nhanh; tiếng Anh + code |
| `jhu-clsp/mmBERT-small` | ~140M | ModernBERT đa ngữ (1800+ ngôn ngữ), mạnh hơn XLM-R; lựa chọn đầu cho tiếng Việt + đa ngữ |
| `microsoft/deberta-v3-small` / `-xsmall` | 44M / 22M backbone | Vẫn rất mạnh cho NLU tiếng Anh |
| `vinai/phobert-base-v2` | 135M | Tiếng Việt, **cần tách từ**, max 256 tokens |
| `uitnlp/visobert` | ~98M | Tiếng Việt mạng xã hội, không cần tách từ |
| `FacebookAI/xlm-roberta-base` | 278M | Baseline đa ngữ an toàn |
| `urchade/gliner_small-v2.1` | ~166M | NER zero-shot với nhãn tùy ý |

### Sentence embedding / retrieval / rerank

| Model | Params | Ghi chú |
|---|---|---|
| `minishlab/potion-base-8M` (Model2Vec) | 8M | Static embedding, chạy CPU cực nhanh; baseline thay TF-IDF |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Baseline kinh điển tiếng Anh |
| `BAAI/bge-small-en-v1.5` | 33M | Tiếng Anh, retrieval tốt so với kích thước |
| `intfloat/multilingual-e5-small` | 118M | Đa ngữ, cần prefix `query: ` / `passage: ` |
| `google/embeddinggemma-300m` | 308M | Top MTEB đa ngữ dưới 500M khi ra mắt (2025); hỗ trợ Matryoshka (cắt 768→128 chiều) |
| `jinaai/jina-embeddings-v5-text-nano` / `-small` | ~239M / ~0.6B | 2026, vendor báo vượt Qwen3-Embedding-0.6B; kiểm tra license |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | Đa ngữ mạnh, có instruction prompt cho query |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22M | Reranker nhỏ tiếng Anh |
| `BAAI/bge-reranker-v2-m3` | 568M | Reranker đa ngữ |
| SetFit (`setfit`) trên embedding nhỏ | — | Few-shot classification (8–64 mẫu/lớp) |

### Small LLM / seq2seq

| Model | Params | Ghi chú |
|---|---|---|
| `google/gemma-3-270m` | 270M | Rất nhỏ, sinh ra để fine-tune task hẹp (extraction, classification dạng sinh) |
| `HuggingFaceTB/SmolLM2-135M/360M-Instruct` | 135M/360M | Tiếng Anh, cực nhẹ |
| `LiquidAI/LFM2-350M` / `-700M` / `-1.2B` | 0.35–1.2B | Kiến trúc hybrid, nhanh trên CPU/edge |
| `Qwen/Qwen3-0.6B` / `-1.7B` / `-4B` | 0.6–4B | Đa ngữ (có tiếng Việt), thinking/non-thinking mode, LoRA dễ |
| `google/gemma-3-1b-it` | 1B | Đa ngữ, context 32k |
| `HuggingFaceTB/SmolLM3-3B` | 3B | Mạnh nhất cỡ 3B open-recipe, long context |
| `google/flan-t5-small` / `VietAI/vit5-base` | 77M / 225M | Seq2seq (tóm tắt, sinh có điều kiện); ViT5 cho tiếng Việt |

**Chiến lược chọn:** TF-IDF/potion-8M baseline (phút) → encoder nhỏ 17–150M fine-tune (task phân loại/NER) → embedding + kNN/LR nếu ít nhãn → small LLM + LoRA chỉ khi task cần sinh văn bản hoặc reasoning. Distillation: train teacher lớn rồi distill logits sang Ettin-32m/MiniLM nếu đề giới hạn latency.

## NLP-14 — Brute force tăng điểm: ensemble + TTA tất tay [CONTEST]

> Dùng khi baseline đã ổn và còn GPU/thời gian. Thứ tự dưới đây xếp theo **gain/công sức** giảm dần. Mọi thứ quyết định trên OOF/val, **không nhìn test**. Nhớ đo tổng thời gian inference xem có vượt giới hạn đề không.

### Pipeline tổng

1. **5-fold CV** (stratified/group) → mỗi fold lưu OOF logits + test logits (`oof_{model}_{seed}.npy`, `test_{model}_{seed}.npy`).
2. **Multi-seed**: 3 seed × 5 fold cho model tốt nhất.
3. **Đa dạng backbone**: 3–5 model khác họ (vd. mmBERT-small, XLM-R-base, PhoBERT-v2, DeBERTa-v3, Qwen3-0.6B + head phân loại) + 1 TF-IDF/char-ngram LR.
4. **Đa dạng input**: max_len 256 vs 512, head-only vs head+tail, có/không tách từ, có/không lowercase → mỗi biến thể là một model mới trong ensemble.
5. **TTA** cho từng model (bảng dưới).
6. **Hill-climbing** chọn model + trọng số trên OOF (code dưới).
7. **Post-processing**: tune threshold từng lớp, calibrate temperature, rule sửa lỗi hệ thống.
8. **Pseudo-label** test/unlabeled bằng ensemble (confidence > 0.9–0.95) → train lại 1 vòng → ensemble lại (chỉ khi luật cho phép).
9. **Full-data retrain**: train lại trên train+val với số epoch tốt nhất, trung bình với các model fold.

### TTA cho NLP

| TTA | Cách làm | Ghi chú |
|---|---|---|
| Multi-window | Văn bản dài: chia sliding window (stride 128), lấy mean/max logits | Gain lớn nhất với text dài |
| Multi-truncation | head / tail / head+tail → trung bình | Rẻ |
| Back-translation TTA | Dịch xuôi-ngược test 1–2 lần, trung bình với bản gốc (trọng số bản gốc lớn hơn) | Chậm; chỉ hữu ích nếu train cũng có BT |
| MC Dropout | `model.train()` khi inference, forward 5–10 lần, lấy trung bình | Hiệu quả ít, nhưng gần như miễn phí |
| Tokenization variants | Có/không tách từ, NFC vs bỏ dấu (nếu train có augment bỏ dấu) | Chỉ khi model đã thấy biến thể lúc train |
| Prompt ensemble (LLM) | 3–5 prompt/few-shot set khác nhau × self-consistency k=5, majority vote | Với LLM zero/few-shot |
| Pair swap | Task cặp câu đối xứng (similarity, paraphrase): `(a,b)` và `(b,a)` | Không dùng cho NLI (không đối xứng) |

### Tricks train thêm điểm

- **Layer-wise LR decay** (0.8–0.95), **re-init top 1–2 layers**, pooling mean+attention thay CLS.
- **EMA/SWA** 30–50% epoch cuối; **AWP/FGM** từ epoch 2; **R-Drop**.
- **Multi-sample dropout** ở head (5 dropout song song, trung bình loss).
- **Distill ensemble → 1 model** nếu inference bị giới hạn thời gian: soft labels của ensemble làm target (KL + CE).
- **Train trên đúng metric**: macro-F1 → class weight/focal loss + tune threshold; ranking → pairwise loss.

```python
# Hill climbing (Caruana) — chọn model có lặp lại, tối đa metric trên OOF
import numpy as np
def hill_climb(oofs: dict, y, metric, iters=50):
    names = list(oofs); best_name = max(names, key=lambda n: metric(y, oofs[n]))
    chosen = [best_name]; cur = oofs[best_name].copy(); best = metric(y, cur)
    for _ in range(iters):
        cand = [(metric(y, (cur * len(chosen) + oofs[n]) / (len(chosen) + 1)), n) for n in names]
        score, n = max(cand)
        if score <= best: break
        chosen.append(n); cur = (cur * (len(chosen) - 1) + oofs[n]) / len(chosen); best = score
    return chosen, best   # trọng số = số lần xuất hiện trong `chosen`
# metric ví dụ: lambda y, p: f1_score(y, p.argmax(1), average="macro")
```

```python
# Sliding-window TTA cho text dài
def predict_long(text, tok, model, max_len=512, stride=128):
    enc = tok(text, truncation=True, max_length=max_len, stride=stride,
              return_overflowing_tokens=True, padding=True, return_tensors="pt")
    enc.pop("overflow_to_sample_mapping", None)
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
    return logits.softmax(-1).mean(0)   # hoặc .max(0) nếu nhãn có thể chỉ xuất hiện 1 đoạn
```

**Cảnh báo:** ensemble > 5 model hiếm khi thêm điểm đáng kể nếu chúng tương quan cao (>0.98 trên OOF) → chọn model đa dạng hơn là thêm seed. Hill-climbing trên val nhỏ dễ overfit → dùng OOF toàn bộ train.
