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
