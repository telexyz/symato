Thu gọn code từ https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo

`./run.sh` để huấn luyện thử (yêu cầu có gpu)

- `dataset.py` load dữ liệu và chuẩn bị dữ liệu huấn luyện dưới dạng `(x, y)`:
  - `x` là chuỗi token đầu vào
  - `y` là token tiếp theo của `x`
- `model.py` rwkv-v4neo model
- `trainer.py` các bước chuẩn bị, điều chỉnh, save / load model params
- `train.py` các kịch bản huấn luyện khác nhau
- `wkv_cuda.*` nhân cuda tăng tốc tính toán khối time-mixing (a.k.a linear attention)
