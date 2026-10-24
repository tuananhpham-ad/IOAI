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
