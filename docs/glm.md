# Pattern-exploiting training (PET)
https://arxiv.org/pdf/2001.07676.pdf
- Tasks có thể đc huấn luyện bở "mô tả" nhúng trong dữ liệu tiền huấn luyện
- Cách này kém hơn supervised learning 
- PET kết hợp 2 ý tưởng trên thành semi-supervised bằng cách biến đổi dữ liệu đầu vào thành cloze-style phrases để giúp mô hình hiểu tác vụ.
- Những cloze-style phrases đó được sử dụng để cho ra những soft labels của một tập lớn những dữ liệu chưa gán nhãn.
- Cuối cùng là quá trình huấn luyện có giám sát (supervised) thông thường
- Nhờ soft labels mà PET thắng đậm supervised training and strong-supervised training trong low-resources.



# Small Language Models Are Also Few-Shot Learners
https://arxiv.org/pdf/2009.07118.pdf

