# Viết inference engine bằng Zig, sử dụng nhiều nhất CPU có thể

Infer engine in Rust: https://github.com/mrsteyk/rwkvk-rs

## Speeding up Python with Zig
https://www.youtube.com/watch?v=O0MmmZxdct4

- Zig là ngôn ngữ lập trình hệ thống hiện đại, ra đời để thay thế C
- Zig hỗ trợ C pointers, import trực tiếp C code và hoàn toàn tương thích với C ABI nghĩa là nó dùng được các C libs có sẵn và C dùng được libs viết từ Zig
- Tại sao không viết extension trực tiếp bằng C?
  - Có những tính năng hiện đại giúp lập trình dễ dàng hơn C
  - Có thư viện chuẩn tốt
  - (sắp) Có trình quản lý gói chính thức
  - Có tool chain để build Zig/C/C++ apps cross flatform
- Chúng ta có thể từng bước cải tiến performance của python code và Zig sẽ giúp làm điều đó

- - -

https://pypi.org/project/setuptools-zig | https://github.com/adamserafini/zaml

```sh
# pip install setuptools-zig ziglang
# PY_VER="python3 -m ziglang"
rm -rf zig-cache/ build/ dist/ zig_sum.egg-info/
PY_VER=/usr/local/bin/zig python3 setup.py bdist_wheel
pip3 install dist/zig_sum-1.0.1-cp310-cp310-linux_x86_64.whl --force-reinstall
python3 -c "from zig_sum import sum; print(sum(20, 22))"
```
