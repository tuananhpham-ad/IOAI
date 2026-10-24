# CV — Học và thực hành với model nhỏ gọn

> Tra cứu nguồn chính thức ngày **2026-10-24**. Ưu tiên model nhỏ nhất đạt chất lượng và runtime yêu cầu. “SOTA” chỉ có nghĩa trong benchmark/task/budget cụ thể; phân biệt model hiện đại với baseline cũ rất nhẹ. Các recipe dưới đây là đề xuất thí nghiệm, chưa có kết quả chạy trong repo.

> Mỗi module cần notebook/script, schema, split, metric, baseline, ablation và phân tích lỗi. Liên quan: [NLP practice](01_NLP_PRACTICE.md), [tuning playbook](04_TUNING_PLAYBOOK.md). P0 = cần thành thạo; P1 = mở rộng sau baseline.

## CV-BS-01 — Dataset và preprocessing đúng (P0)

1. Xác định task: classification, multi-label, detection, semantic/instance segmentation, retrieval hay image-text. Chốt metric trước: accuracy/macro-F1, mAP50–95, mIoU/Dice, Recall@k hoặc task accuracy.
2. Audit ảnh hỏng, EXIF orientation, RGB/BGR, grayscale/alpha, bit depth; vẽ ảnh kèm nhãn. BBox cần rõ `xyxy`, `xywh` hay YOLO normalized; mask cần class IDs, background và ignore index.
3. Tìm exact/near duplicates, chia theo người/vật thể/video/camera/site/thời gian khi cần. Frame liền nhau và crop của cùng ảnh ở cùng split.
4. Resize/crop/pad theo checkpoint và task; normalize đúng mean/std/range từ processor/weights. Validation dùng transform xác định, không random train augmentation.
5. Dataset-specific statistics chỉ tính từ train. Giữ metadata scale/pad/crop để đổi tọa độ dự đoán về ảnh gốc.

| Dữ liệu / task | Preprocess nên so | Bẫy cần kiểm tra |
|---|---|---|
| Classification vật thể lớn | Resize + center crop baseline; RandomResizedCrop khi train | Crop mất vật thể nhưng vẫn giữ label; thử pad/resize giữ toàn ảnh nếu nhãn ở rìa |
| Detection | Letterbox giữ tỷ lệ; pad theo stride; cập nhật boxes | Đổi RGB/BGR, box format hoặc quên bỏ padding lúc trả output |
| Semantic segmentation | Resize ảnh bilinear/bicubic; mask nearest-neighbor | Nội suy mask tạo class ID giả; background/ignore index khác quy ước checkpoint |
| Vật thể/khuyết tật nhỏ | Tiling có overlap, crop ROI, độ phân giải cao có kiểm soát | Resize xuống làm mất tín hiệu; chia split trước tiling; gộp dự đoán vùng overlap |
| OCR, chữ/số, trái/phải | Giữ orientation và aspect ratio; pad thay vì ép méo | Horizontal flip/rotation có thể đổi chữ, số hoặc nhãn hướng |
| Màu quyết định nhãn | Giữ color space; thử white balance nhẹ nếu có domain shift | Grayscale, hue jitter mạnh có thể xóa tín hiệu |
| Medical/satellite/16-bit | Dùng domain range/window theo task; kiểm tra số kênh | Ép ảnh 16-bit xuống 8-bit hoặc dùng ImageNet normalization tùy tiện |

Nguồn triển khai: [torchvision transforms v2](https://docs.pytorch.org/vision/stable/transforms.html), [Albumentations: bbox transforms và tọa độ](https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/), [timm model/data configuration](https://huggingface.co/docs/timm/main/model_api). Pin version vì API thư viện thay đổi; đọc config checkpoint thay vì hard-code chung cho mọi model.

**Lab `LAB-CV-01`:** tạo contact sheet 32 ảnh trước/sau preprocess, overlay boxes/masks; kiểm tra label map và min/max tensor. So resize/crop với letterbox hoặc tiling trên cùng split, đo metric và latency. Không tăng resolution nếu không đủ thời gian inference.

## CV-BS-02 — Augmentation theo bất biến của nhãn (P0)

Chọn biến đổi mô phỏng điều kiện có thể gặp ở test. Chỉ augment train; TTA ở CV-BS-06 là một pipeline inference riêng. Các khoảng sau là điểm bắt đầu, không phải công thức chung cho mọi dataset.

| Kỹ thuật | Khởi đầu để thử | Hợp với / rủi ro |
|---|---|---|
| Horizontal flip | `p=0.5` | Ảnh tự nhiên nếu nhãn không phụ thuộc trái/phải; tránh OCR/biển chỉ hướng |
| RandomResizedCrop | `scale=(0.7,1.0)`, ratio gần ảnh gốc | Classification; crop quá nhỏ làm mất nhãn, đặc biệt ảnh nhiều đối tượng |
| Affine/rotation | ±5–10°, translate ≤5% | Lệch góc nhẹ; cập nhật boxes/masks/keypoints đồng bộ |
| Color jitter | brightness/contrast/saturation khoảng 0.1–0.2 | Ánh sáng thay đổi; giảm hoặc bỏ nếu màu là nhãn |
| Blur/noise/JPEG | Một loại/lần, `p=0.1–0.2`, mức nhẹ | Camera/OCR noisy; đánh giá riêng clean/noisy; không phá chi tiết nhỏ |
| Random erasing / Cutout | `p=0.1–0.25`, vùng xóa nhỏ | Occlusion robustness; đừng che hết vật thể; xử lý target theo task |
| MixUp | `α=0.1–0.4`, bắt đầu classification | Trộn ảnh và soft labels; criterion phải nhận soft targets |
| CutMix | `α≈1`, ablate riêng với MixUp | Ghép patch; label weight dựa trên diện tích thực sau clip box, không dùng λ chưa điều chỉnh |
| RandAugment / TrivialAugmentWide | Policy nhẹ trước, kiểm tra ảnh mẫu | Ít tune từng phép; danh sách phép có thể chứa biến đổi làm sai nhãn domain |
| AugMix | P1 cho corruption robustness | Nhiều nhánh augment; đo thêm train cost và consistency objective nếu dùng |
| Mosaic | Detection; bật có kiểm soát, thử tắt cuối train | Vật thể nhỏ/nhiều bối cảnh; tạo phân phối và scale bất thường |
| Copy-Paste | Có instance masks tốt; paste ảnh + mask + target | Tăng đối tượng hiếm; xử lý occlusion/overlap và background hợp lý |
| Class-aware augmentation | Ưu tiên lớp thiếu, cap số bản/mẫu | Không giải quyết nhãn sai; oversample mạnh có thể overfit ảnh gốc |

MixUp: `x = λ*x_a + (1-λ)*x_b`, `y = λ*y_a + (1-λ)*y_b`. Đây là soft-label classification; detection/segmentation cần cách trộn targets riêng. Không cộng mạnh MixUp, CutMix và label smoothing ngay từ đầu vì có thể underfit. Xem [hướng dẫn MixUp/CutMix của torchvision](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html), [AugMix paper](https://arxiv.org/abs/1912.02781) và [Mosaic/Copy-Paste trong Ultralytics](https://docs.ultralytics.com/guides/yolo-data-augmentation/).

**Ba recipe khởi đầu:**

- **Classification ít data:** resize/crop giữ vật thể + flip hợp lệ → thêm jitter nhẹ → thử riêng MixUp hoặc CutMix. Với model frozen, precompute nhiều view nếu cần augment; một embedding/ảnh cố định không tạo biến thể mới mỗi epoch.
- **Detection:** letterbox + flip/scale nhẹ → Mosaic → Copy-Paste khi có masks; trực quan ít nhất 32 transformed targets trước train dài.
- **Segmentation:** spatial transform chung ảnh/mask → photometric transform chỉ trên ảnh; mask vẫn là integer class IDs. Với keypoints có nhãn trái/phải, flip cần cả hoán đổi semantic labels.

**Lab `LAB-CV-02`:** A0 không augment → A1 hình học nhẹ → A2 thêm photometric → A3 MixUp **hoặc** CutMix. Giữ số steps/data exposure tương đương; ghi metric theo lớp/kích thước vật thể và clean/noisy val. Chỉ ghép các phép đã có ích.

## CV-BS-03 — Shortlist model nhỏ theo task (P0)

### Classification và feature extraction

M = triệu tham số. Params trong bảng có thể gồm classifier ImageNet, còn backbone-only không gồm head; đo lại sau khi đổi số classes. Thứ tự dưới đây ưu tiên kích thước, không phải bảng xếp hạng accuracy.

| Model / nguồn | Params ≈ | Vai trò | Điểm cần nhớ |
|---|---:|---|---|
| [MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) | 2.54M | Baseline cũ rất nhẹ | Bắt đầu CPU/edge, frozen head rồi fine-tune; weights API có transform riêng |
| [MobileNetV4 Conv Small](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k) | 3.8M | CNN mobile hiện đại | ID `mobilenetv4_conv_small.e2400_r224_in1k`; card ghi train 224, test 256; lấy eval transform từ config |
| [RepViT-M0.9](https://github.com/THU-MIG/RepViT) | 5.1M | CNN tối ưu triển khai | Có reparameterization cho deploy; “M0.9” là tên theo latency công bố, không phải 0.9M params |
| [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | 21.6M | Feature backbone mới, frozen probe/retrieval | Không có classifier cho nhãn riêng; bắt đầu frozen features; model có điều khoản truy cập/license riêng |
| [MobileCLIP2-S0](https://github.com/apple-aiml-research/ml-mobileclip) | 11.4M image + 63.4M text ≈ 74.8M | Zero-shot image-text / retrieval | Nếu labels cố định, cache text vectors; chỉ bỏ text encoder khi không cần query text mới |

**Bằng chứng có phạm vi:** repo Apple báo MobileCLIP2-S0 đạt 71.5% ImageNet-1k zero-shot top-1. Đây là đánh giá image-text, không so trực tiếp với supervised ImageNet top-1 của MobileNet/RepViT. Latency trên thiết bị tác giả không phải latency máy mình. Nguồn: [bảng MobileCLIP2](https://github.com/apple-aiml-research/ml-mobileclip).

**Model mới chưa chắc nhỏ:** [MobileNetV5-300M Gemma3n](https://huggingface.co/timm/mobilenetv5_300m.gemma3n) là image encoder khoảng 0.3B tham số. Không tự thay MobileNetV4 3.8M bằng nó khi mục tiêu là footprint tối thiểu.

### Detection, segmentation, image-text generation

| Model / nguồn | Params ≈ | Task đúng | Khi sử dụng |
|---|---:|---|---|
| [YOLO26n](https://docs.ultralytics.com/models/yolo26/) | 2.4M sau fuse/loại nhánh inference không dùng | Object detection | Ứng viên detector mới rất nhỏ; checkpoint train có thể đếm lớn hơn; không lấy số này cho mọi task suffix |
| [SegFormer-B0 ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) | Khoảng 3.7M cả segmentation head | Semantic segmentation | Head có nhãn ADE20K; đổi head/label map khi fine-tune dataset riêng |
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Khoảng 9.66M cả pipeline | Promptable segmentation | Cần point/box prompts; mask không tự mang semantic class; không thay trực tiếp semantic segmenter |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | LM 800M + vision encoder | VQA / image-text generation | Chỉ dùng khi cần ngôn ngữ; lớn hơn nhiều so CNN; đo số visual tokens và output tokens |

**YOLO26n:** docs công bố COCO val mAP50–95 40.9 và mAP end-to-end 40.1 ở input 640; phân biệt hai chế độ khi so sánh. Params/FLOPs trong bảng là sau fuse và bỏ nhánh không dùng, không phải nguyên checkpoint training. Các số này là của tác giả, chưa đo trong repo. Xem [evaluation và ghi chú chính thức](https://docs.ultralytics.com/models/yolo26/).

**Không đủ một bảng params để quyết định:** CNN ít params có thể bị chậm bởi memory access; attention phụ thuộc resolution; segmentation có activation lớn. Báo `artifact_mb`, peak RAM/VRAM, p50/p95 latency, throughput, input size, batch, dtype/backend và metric cùng nhau. Số weights FP16 lý tưởng ≈ `2 × params` bytes, không phải toàn bộ VRAM.

## CV-BS-04 — Transfer learning và fine-tuning tiết kiệm (P0)

1. **Baseline:** classifier đơn giản trên frozen pretrained features; thử linear/k-NN. Với ảnh có tính chất đơn giản, thêm histogram/HOG + linear để biết deep model có thực sự cần thiết.
2. **Head-only:** freeze backbone; giữ BatchNorm running statistics cố định nếu không chủ ý thích nghi; `requires_grad=False` không tự đặt BatchNorm về eval.
3. **Partial unfreeze:** mở stage cuối rồi mới full FT; learning rate backbone thấp hơn head là ứng viên cần đo.
4. **Training:** CE single-label; BCEWithLogits multi-label; segmentation xử lý ignore index, imbalance và Dice/CE đúng format; detection ưu tiên recipe chính thức của framework.
5. **Regularization:** nhẹ trước; EMA khi framework hỗ trợ; class weights/sampler chỉ từ train; tránh vừa oversample mạnh vừa tăng loss weight mạnh không ablate.
6. **Efficiency:** mixed precision phù hợp thiết bị; batch/accumulation theo memory; tăng resolution sau khi baseline đúng. Accumulation không tương đương batch lớn đối với BatchNorm.

| Tham số | Candidate thực hành | Điều kiện |
|---|---|---|
| AdamW head LR | `[1e-4, 3e-4, 1e-3]` | Không copy sang mọi optimizer |
| AdamW backbone LR | `[1e-5, 3e-5, 1e-4]` | Partial/full FT, theo loss stability |
| Weight decay | `[0.01, 0.05]` | Xử lý norm/bias theo recipe |
| Image size | `[160,224,256]` cho classification nếu model cho phép | Detection/segmentation dùng size/stride riêng; xem small-object recall |
| Augmentation | None / light / một policy mạnh | Chưa gộp nhiều policy ở lượt đầu |
| Train duration | Head-only vài epoch rồi quyết định unfreeze | Chọn checkpoint bằng val, không theo train loss |

**Lab `LAB-CV-03`:** MobileNetV3-Small → MobileNetV4-Small, so head-only và partial FT. Thêm DINOv3 frozen probe nếu có sẵn weights được phép dùng. Báo trainable **và total** params, quality/runtime; không bắt buộc model mới thắng.

## CV-BS-05 — Thực hành detection và segmentation (P0/P1)

### Detection: `LAB-CV-04`

- Baseline YOLO26n, đọc data YAML/class IDs và chạy pretrained inference hoặc fine-tune tùy label set.
- Đánh giá mAP50–95, AP theo class/size; lưu false positives/false negatives; chỉ báo mAP50 là chưa đủ nếu scorer dùng IoU 0.50:0.95.
- Ablate resolution → spatial augmentation → Mosaic. Với vật thể nhỏ, thử tiling rồi merge boxes về ảnh gốc; tính cả thời gian crop/merge.
- Tune confidence cho operating point F1/precision-recall riêng. Khi đo AP cần đủ prediction scores cho PR curve; đừng cắt confidence quá cao trước scorer.
- YOLO26 có đường inference end-to-end; không gắn thêm NMS mặc định nếu backend đã cung cấp output cuối. Ensemble/TTA cần fusion riêng và xác minh export semantics.

### Segmentation: `LAB-CV-05`

- Baseline SegFormer-B0 hoặc encoder nhỏ + decoder phù hợp; log **tổng encoder+decoder** params, không chỉ backbone.
- Kiểm tra foreground/background, ignore index và xử lý ảnh không có foreground. Metric mIoU/Dice cần quy ước nhất quán khi một lớp vắng mặt.
- Resize **logits** về resolution cần chấm trước `argmax`; mask class IDs dùng nearest-neighbor khi cần transform.
- Ablate patch sampling foreground, spatial augmentation, CE/Dice và overlap inference; tránh chỉ lấy patches có foreground gây lệch phân phối train.
- MobileSAM là lab riêng về point/box prompt: đo IoU và số prompts, không dùng mask ground truth làm prompt lúc test nếu task không cho phép.

## CV-BS-06 — Ensemble và test-time augmentation (P1)

| Cách | Thao tác | Lưu ý chi phí / tính đúng |
|---|---|---|
| Classification soft voting | Average probabilities MobileNet + RepViT hoặc 2 seeds; thử weights đều trước | Căn sample IDs/labels; softmax cho single-label, sigmoid cho multi-label |
| Classification TTA | Original + horizontal flip nếu hợp lệ; trung bình probabilities | Gần gấp đôi forward; five-crop/multi-scale đắt hơn; crop phải còn evidence |
| Segmentation TTA | Flip/scale ảnh → dự đoán → inverse transform → average aligned maps | Bỏ pad/đưa về cùng resolution trước blend; không average mask IDs đã argmax |
| Detection TTA/ensemble | Đổi boxes về tọa độ ảnh gốc → class-wise fusion | WBF/NMS/Soft-NMS cần tune IoU/confidence trên val; không average box theo index |
| Seed/fold ensemble | 2 seeds hoặc 3 folds trước | Train/infer cost tăng; folds phải giữ group split |
| EMA/SWA/soup | Average weights tương thích, deploy một model | Không average khác architecture; BatchNorm statistics có thể cần tính lại trên train |
| Cascade | Model nhỏ xử lý tất cả, model mạnh xử lý low-confidence subset | Tune routing trên val; đo cả latency trung bình và worst case; confidence lệch domain có thể bỏ sót lỗi |

**WBF** dùng confidence để gộp các boxes tương ứng; mọi input cần cùng class map, hệ tọa độ và normalization. Nó không bảo đảm tốt hơn NMS trên mọi dataset. Nguồn: [Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302).

**Soup/SWA không phải prediction ensemble:** model soup cần các lần fine-tune tương thích, thường từ cùng pretrained initialization; chọn soup trên val và giữ model tốt nhất nếu average làm giảm điểm. Nguồn: [Model soups](https://arxiv.org/abs/2203.05482).

**Lab `LAB-CV-06`:** single best → + flip TTA → + model thứ hai → EMA/soup nếu có checkpoint hợp lệ. Xuất bảng metric/latency/memory; chỉ giữ phương án nằm trong budget. Training-only augmentation không làm tăng số forward inference, còn TTA có tăng.

## CV-BS-07 — Giữ chất lượng nhưng giảm model triển khai (P1)

- **Distillation:** teacher mạnh hoặc ensemble → MobileNetV3/V4 student. Thử CE + temperature-scaled KL cho classification; feature distillation cần adapter khi channel/spatial size khác; segmentation/detection cần mask/box alignment và objective riêng. Teacher chỉ sinh targets từ data train/unlabeled được phép. Nguồn: [Knowledge distillation](https://arxiv.org/abs/1503.02531).
- **PTQ INT8:** calibration trên train đại diện; so metric FP32/FP16/INT8 sau export. Nếu giảm chất lượng đáng kể, thử QAT với backend hỗ trợ; model nhỏ vẫn cần kiểm tra latency thực.
- **Reparameterization / fusion:** dùng deploy conversion chính thức của RepViT hoặc Conv-BN fusion khi hợp lệ; so output trước/sau. Không chuyển deploy rồi tiếp tục train như cấu trúc ban đầu.
- **Resolution/crop:** giảm input thường giảm compute rõ hơn giảm vài tham số; kiểm tra small-object recall trước khi chọn. Progressive resizing chỉ dùng nếu pipeline/checkpoint hỗ trợ.
- **Cache:** frozen features; MobileCLIP cache text embeddings cho labels cố định; không giữ text encoder nếu inference thực sự chỉ cần ảnh và vector lớp đã chuẩn bị.
- **Structured pruning:** chỉ đáng làm khi backend tận dụng shape mới; unstructured sparsity không tự làm model nhanh. Export artifact phải chứa weights cần dùng và preprocessing/postprocessing đúng.

**Exit:** reload artifact, so predictions trước/sau export theo tolerance phù hợp dtype, đo end-to-end gồm decode ảnh + preprocess + model + postprocess. Chạy batch=1 và batch thực tế; warm-up và đồng bộ GPU; ghi hardware, threads, backend versions. Không lấy latency chỉ của forward làm tổng thời gian nộp bài.

## CV-BS-08 — Lộ trình chọn nhanh và bảng ablation

| Ràng buộc | Bắt đầu | Nâng cấp khi val chứng minh cần |
|---|---|---|
| Classification, tối thiểu params | MobileNetV3-Small 2.54M | MobileNetV4-Small 3.8M → RepViT-M0.9 5.1M |
| Ít nhãn, features tổng quát | Frozen small backbone + linear | DINOv3 ViT-S/16 frozen; partial FT sau |
| Zero-shot nhãn bằng text | MobileCLIP2-S0 + prompt templates | Cache/average text embeddings; thêm prompt chỉ khi val tốt hơn |
| Detection nhỏ gọn | YOLO26n | Resolution/augmentation trước model lớn hơn |
| Semantic segmentation | SegFormer-B0 | Sampling/loss/tiling trước backbone lớn |
| Promptable mask | MobileSAM | Tối ưu prompts và cache image embeddings |
| VQA/OCR reasoning có output text | Qwen3.5-0.8B nếu budget cho phép | Giới hạn image resolution/tokens, tune prompt và format trước |

**Thứ tự thí nghiệm:** split + scorer → preprocess → baseline nhỏ → augmentation → fine-tune → ensemble/TTA → distill/quantize. Mỗi run ghi `dataset_hash, split, seed, model_revision, input_size, augment_config, metric, total_params, trainable_params, artifact_mb, train_seconds, inference_p50/p95, peak_ram/vram, backend`.

**Điều kiện dừng:** model nhỏ đạt metric tối thiểu và inference budget; nâng cấp không cải thiện vượt variance; hoặc không đủ thời gian đóng gói/reload. Khi score chênh nhỏ, xác nhận thêm seeds cho finalist thay vì mở thêm nhiều model.

## CV-BS-09 — Liên hệ bài trong workspace

- [Double Agent Dilemma](example/4_Double_Agent_Dilemma/README.md): đọc scorer và statement trước; bài khai thác khác biệt giữa ResNet18 và ViT-Tiny. Ensemble giảm bất đồng có thể đi ngược mục tiêu bài; dùng để học architecture bias và phân tích lỗi, không áp recipe classification máy móc.
- README [Individual Contest](example/README.md) mô tả `IOAI Field` train generator từ đầu, không dùng pretrained weights. Shortlist pretrained trong tài liệu này chỉ áp dụng khi đề cho phép; task đó chưa có thư mục tương ứng trong workspace hiện tại.

**Artifacts cần nộp cho mỗi lab:** config + notebook/script, `metrics.json`, bảng ablation, ảnh lỗi/overlay, checkpoint hoặc hướng dẫn tải đúng revision, và lệnh inference offline chạy lại được. Không điền số đo giả vào bảng.

## CV-07 — Brute force tăng điểm: ensemble + TTA tất tay [CONTEST]

> Thứ tự theo gain/công sức. Quyết định trên OOF/val, không nhìn test; đo tổng thời gian inference.

### Pipeline tổng

1. **5-fold CV** (group split nếu có), lưu OOF + test probabilities cho từng model/seed.
2. **Tăng resolution**: train lại ở res cao hơn (224→384→448/518; detection 640→1024/1280). Thường là gain lớn nhất với vật thể nhỏ/chi tiết.
3. **Đa dạng kiến trúc**: 1–2 CNN (ConvNeXtV2, EfficientNetV2) + 1–2 ViT (DINOv2/v3, EVA02, SigLIP2 image encoder) + 1 linear probe DINO/SigLIP.
4. **Multi-seed** cho model tốt nhất; **EMA** luôn bật.
5. **TTA** (bảng dưới) cho mọi model.
6. **Hill-climbing / weight optimize** trên OOF (dùng lại code ở NLP-14).
7. **Pseudo-label** test/unlabeled bằng ensemble (soft labels hoặc confidence cao) → train lại → ensemble lại (nếu luật cho phép).
8. **Full-data retrain** + **knowledge distillation** nếu bị giới hạn latency.
9. **Post-processing**: threshold từng lớp, WBF cho detection, remove small components/fill holes cho segmentation.

### TTA cho CV

| Task | TTA | Gộp kết quả |
|---|---|---|
| Classification | hflip, vflip (nếu ảnh không có hướng: y tế, vệ tinh), rot90×4, multi-scale (0.9/1.0/1.15), five-crop/ten-crop | Trung bình xác suất (hoặc logits) |
| Segmentation | flip + multi-scale, **đảo ngược biến đổi trên mask** trước khi gộp; sliding window với overlap cho ảnh lớn | Trung bình xác suất pixel → threshold |
| Detection | hflip + multi-scale (`ultralytics`: `model.predict(augment=True)`), tiling SAHI cho vật nhỏ | **WBF**, không NMS (NMS bỏ box, WBF trộn box) |
| Keypoint | hflip phải **hoán đổi cặp điểm trái/phải** | Trung bình heatmap |
| Ít data / feature model | Embedding của nhiều crop/flip trung bình rồi mới kNN/LR | |

Thư viện: `ttach` (classification/seg TTA có sẵn), `ensemble-boxes` (WBF), `sahi` (sliced inference).

```python
import ttach as tta
tf = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[0, 90, 180, 270]),
                  tta.Scale(scales=[1, 1.15])])
seg_tta = tta.SegmentationTTAWrapper(model, tf, merge_mode="mean")   # tự đảo ngược mask
cls_tta = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform(), merge_mode="mean")
```

```python
# Detection: TTA + nhiều model → WBF
from ensemble_boxes import weighted_boxes_fusion
B, S, L = [], [], []
for m in models:                     # vd. yolo26s, yolo26m, rfdetr-small
    for flip in [False, True]:
        img_in = img[:, ::-1] if flip else img
        r = m.predict(img_in, conf=0.001, verbose=False)[0]   # conf thấp, để WBF lọc
        b = r.boxes.xyxyn.cpu().numpy()
        if flip: b[:, [0, 2]] = 1 - b[:, [2, 0]]
        B.append(b); S.append(r.boxes.conf.cpu().numpy()); L.append(r.boxes.cls.cpu().numpy())
boxes, scores, labels = weighted_boxes_fusion(B, S, L, iou_thr=0.6, skip_box_thr=0.001,
                                              conf_type="avg")
```

### Tricks train thêm điểm

- **Fine-tune 2 giai đoạn**: freeze backbone train head → unfreeze với layer-wise LR decay.
- **Mixup/CutMix + label smoothing + drop path** cho ViT; tắt augment mạnh ở 10–20% epoch cuối (giống `close_mosaic`).
- **Progressive resizing**: train res thấp → fine-tune vài epoch res cao.
- **Loss theo metric**: focal/Lovász/Dice cho segmentation IoU, weighted CE cho macro-F1.
- **Train ở res cao + test ở res cao hơn một chút** (FixRes: test res ≈ 1.15× train res thường lợi).

**Cảnh báo:** rot90/vflip TTA làm *giảm* điểm với ảnh có hướng tự nhiên (người, xe, chữ) → bật từng TTA và đo trên val. Kiểm tra tương quan OOF trước khi thêm model; ưu tiên đa dạng kiến trúc + resolution hơn thêm seed.
