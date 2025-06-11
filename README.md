# Football Player Classification

## Mục tiêu
Dự án Football Player Classification được xây dựng nhằm mục tiêu phát triển một hệ thống sử dụng kỹ thuật học sâu (Deep Learning) để phân loại cầu thủ bóng đá dựa trên hình ảnh hoặc video. Thông qua quá trình huấn luyện mô hình trên tập dữ liệu gồm các hình ảnh/video của các cầu thủ với nhãn tương ứng (như tiền đạo, hậu vệ, thủ môn,...), hệ thống có thể tự động nhận diện và phân loại cầu thủ trong các tình huống thực tế trên sân bóng. Dự án hướng đến việc hỗ trợ các ứng dụng như phân tích trận đấu, huấn luyện chiến thuật, hoặc làm nền tảng cho các hệ thống phân tích thể thao thông minh.
```yaml

## Cấu trúc dự án
Football_Player_Classification/
├── generate_dataset.py # Tạo bộ dữ liệu từ hình ảnh/video gốc
├── FootballDataset.py # Dataset class cho PyTorch
├── train_football.py # Huấn luyện mô hình classification
├── inference_image.py # Dự đoán trên ảnh đơn
├── inference_video.py # Dự đoán trên video

```

## Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/nd-wuangr26/Football_Player_Classification.git
cd Football_Player_Classification
```

## Huấn luyện mô hình
```
python train_football.py \
  --data_dir data/ \
  --epochs 20 \
  --batch_size 16 \
  --lr 0.001
```
## Dự đoán trên video
```
python inference_video.py \
  --model_path best_model.pth \
  --video_path input.mp4 \
  --output_path output.mp4
```
## Dự đoán trên ảnh
```
python inference_image.py \
  --model_path best_model.pth \
  --image_path path/to/image.jpg
```
