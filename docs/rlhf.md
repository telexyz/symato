# Illustrating Reinforcement Learning from Human Feedback
https://huggingface.co/blog/rlhf

- Đào tạo trước một mô hình ngôn ngữ (LM),
- thu thập dữ liệu và đào tạo một mô hình khen thưởng, và
- tinh chỉnh LM với học tăng cường.

## Reward model training
![](files/rlhf-00.png)

## Fine-tuning with RL

 Tinh chỉnh một số hoặc tất cả các tham số của bản sao của LM ban đầu bằng policy-gradient RL algorithm, Tối ưu hóa Chính sách Gần nhất (PPO: Proximal Policy Optimization). Các tham số của LM bị đóng băng vì việc tinh chỉnh toàn bộ mô hình tham số 10B hoặc 100B+ rất tốn kém (để biết thêm thông tin, hãy xem Thích ứng với thứ hạng thấp (LoRA) cho LM hoặc Sparrow LM từ DeepMind).

Đầu tiên chúng ta hãy xây dựng nhiệm vụ tinh chỉnh này như một bài toán RL. Đầu tiên, chính sách là một mô hình ngôn ngữ nhận lời nhắc và trả về một chuỗi văn bản (hoặc chỉ là phân phối xác suất trên văn bản). __Không gian hành động__ của chính sách này là tất cả các tokens trong bộ từ vựng của mô hình ngôn ngữ (khoảng 50k) và __không gian quan sát__ là các chuỗi tokens đầu vào có thể, cũng khá lớn (vocab_size ^ tokens_len). __Hàm phần thưởng__ là sự kết hợp của preference model và một ràng buộc đối với sự thay đổi chính sách.

Hàm phần thưởng là nơi hệ thống kết hợp tất cả các mô hình mà chúng ta đã thảo luận thành một quy trình RLHF. Khi nhận được lời nhắc, x, từ tập dữ liệu, hai văn bản, y1, y2, sẽ được tạo – một từ mô hình ngôn ngữ ban đầu và một từ lần lặp lại hiện tại của chính sách tinh chỉnh. Văn bản từ chính sách hiện tại được chuyển đến mô hình ưu tiên, mô hình này trả về một khái niệm vô hướng về “mức độ ưu tiên”, `r_theta`. Văn bản này được so sánh với văn bản từ mô hình ban đầu để tính toán một hình phạt về sự khác biệt giữa chúng. Trong nhiều bài báo từ OpenAI, Anthropic và DeepMind, hình phạt này đã được thiết kế như một phiên bản thu nhỏ của phân kỳ Kullback–Leibler (KL) giữa các chuỗi phân phối này qua tokens, `r_{KL}`.

`r_{KL}` "phạt" chính sách RL nếu nó di chuyển đáng kể khỏi mô hình cho trước từ đầu qua mỗi mẻ đào tạo, điều này có thể hữu ích để đảm bảo mô hình xuất ra các đoạn văn bản mạch lạc hợp lý. Nếu không có hình phạt này, quá trình tối ưu hóa có thể bắt đầu tạo ra văn bản vô nghĩa nhưng lại đánh lừa mô hình phần thưởng để mang lại phần thưởng cao. Trong thực tế, phân kỳ KL được tính gần đúng thông qua lấy mẫu từ cả hai bản phân phối ([giải thích tại đây](http://joschu.net/blog/kl-approx.html)). Phần thưởng cuối cùng được gửi đến quy tắc cập nhật RL là `r = r_theta - lambda r_{KL}`.

Cuối cùng, quy tắc cập nhật là bản cập nhật tham số từ PPO nhằm tối đa hóa chỉ số phần thưởng trong lô dữ liệu hiện tại (PPO  is on-policy, nghĩa là các tham số chỉ được cập nhật với current batch of prompt-generation pairs). PPO là thuật toán tối ưu hóa vùng tin cậy sử dụng các ràng buộc về gradient để đảm bảo bước cập nhật không làm mất ổn định quá trình học. DeepMind đã sử dụng một thiết lập phần thưởng tương tự cho Gopher nhưng đã sử dụng lợi thế đồng bộ tác nhân-phê bình (A2C: [advantage actor-critic](proceedings.mlr.press/v48/mniha16.html)) để tối ưu hóa gradient, which is notably different but has not been reproduced externally.

![](files/rlhf-01.png)

# Fine-Tuning Language Models from Human Preferences
https://arxiv.org/pdf/1909.08593.pdf

# LoRA: Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685


# Offline RL for Natural Language Generation with Implicit Language Q Learning
https://sea-snell.github.io/ILQL_site


# Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
https://arxiv.org/pdf/2204.05862.pdf


# Exploring the Benefits of Training Expert Language Models over Instruction Tuning
- https://github.com/joeljang/ELM
- https://arxiv.org/pdf/2302.03202.pdf

- - -

# TRL - Transformer Reinforcement Learning
https://github.com/lvwerra/trl
