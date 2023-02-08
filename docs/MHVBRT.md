https://www.facebook.com/1002327458/posts/pfbid02HhgKLQDzq1iscL3XXMHzqoPdBeZVZsyfm2tyDkYMjr1EmmRa9WYVbonchWbSSERYl

# MÔ HÌNH VĂN BẢN RẤT TO (MHVBRT)

Mô-hình-ngôn-ngữ-lớn là thuật ngữ mới nổi mấy năm nay, dùng để nói về những mô hình xác suất huấn luyện trên dữ liệu văn bản cực lớn. Để hình dung, mô hình GPT-3 được huấn luyện trên 500 tỉ từ. Giả sử mỗi phút con người chúng ta đọc được 250 từ, một ngày 5 tiếng miệt mài suốt cả năm, thì cần mất 18 ngàn năm để đọc từng ấy từ. Cỡ bằng thời gian Hành Giả Tôn tu luyện từ đá mà thành.

Đại để, các mô hình này cho ta một ước tính định lượng về xác suất một đoạn văn có thể tồn tại. Xác suất càng lớn thì đại khái đoạn văn nghe càng mượt mà, y như đại đa số chúng ta viết ra. Chính vì cái lẽ mượt mà đó mà mô hình ngôn ngữ được dùng bất cứ khi nào cần sinh ra đoạn văn hợp nhĩ, như trong dịch thuật, nhận dạng tiếng nói, thêm dấu tiếng Việt hay khi cần đánh giá chất lượng văn bản.

Tất nhiên "hợp nhĩ" không có nghĩa là nó đúng, vì thế nó có thể phịa ra đủ thứ nghe như thật Nên nếu cô nào nghe được một câu rất mượt như rót mật vào tai, hẳn chẳng phải là máy nó nghĩ thế thật. Đơn giản là nó mix từ văn vở của những anh chàng dẻo mỏ trên cõi mạng thôi.

Cho nên muốn nó nói ra cái gì có nghĩa, thì phải tổ lái. Dân dã gọi là __mớm cung__, sang miệng gọi là "prompt engineering", dân thống kê gọi là mô hình ngôn ngữ có điều kiện. Tới đây kỹ năng quan trọng bậc nhất chính là tổ lái này. Nghe đâu có nơi trả lương tới mấy trăm ngàn đô/năm cho những anh lái thực dẻo.

Cụ Chomsky khá là bức xúc với khái niệm mô-hình-ngôn-ngữ này. Với cụ, chúng chẳng phải là mô hình ngôn ngữ gì sất, vì nó chẳng cho cụ biết thêm gì về ngôn ngữ. Ngôn ngữ theo cụ, phải là cái gì tinh túy trời sinh ra thế, có cú pháp sinh vô hạn, chứ không phải là con số vô tri. Cụ gì đấy lại nói, chúng chỉ là những mô hình văn bản. Thôi thì nể mặt các cụ, nhà cháu gọi là mô-hình-văn-bản-rất-to (MHVBRT).

## Vậy MHVBRT to thế nào? 
Rất to so các mô hình xử lý ảnh phổ biến như ResNet-101, có tầm khoảng 50 triệu tham số. GPT-3 có 175 tỉ tham số, tức tầm 3-4 nghìn lần. Nhưng lại rất bé so với não người có khoảng 1.8-3.2 trăm nghìn tỉ kết nối. Vậy nên còn một quãng đường rất dài để GPT có thể khôn như người. Nói vậy không có nghĩa là cách thức tạo nên não người thông qua tiến hóa là hợp lý nhất, và trí khôn không nhất thiết phải như người. Và không nên đồng nhất năng lực văn bản và trí khôn, nếu như câu chuyện chỉ là sinh ra văn bản có xác suất cao.

Vì sao cần đến 175 tỉ tham số để mô tả văn bản? Là vì ngôn ngữ cần vài chục ngàn từ vựng để mô tả thế giới và ý nghĩ, lại khá nhập nhằng ngữ nghĩa. Có vô hạn cách diễn đạt cùng một ý. Nhưng nếu không gian ngôn ngữ là vô hạn, thì vì sao 175 tỉ tham số lại khá ổn cho công việc hàng ngày? Là vì không gian hiệu dụng thực ra lại rất nhỏ. Nói đi nói lại cũng có từng ấy ý, mà thường thì phải nói theo cùng một khuôn mẫu để người khác không hiểu nhầm.
Sự lặp lại ấy cho phép ta nén thông tin. MHVBRT là một máy nén. Ví dụ, GPT-3 nén 500 tỉ từ vào 175 tỉ tham số. Giả thử mỗi từ cho chúng ta 16 bit thông tin, và mỗi tham số diễn đạt được 3 bit, thì tỉ suất nén là 500 * 16 / (3*175) ~ 15. Có lẽ có thể nén được thêm vài chục lần nữa, ít nhất cho các trường hợp riêng.

Máy nén này sẽ bung ra văn bản dựa trên những câu ta mớm cho nó. Thi thoảng nó sẽ bung ra y chóc câu nó học. Không phải là cố tình làm thế, mà có thể vì câu mớm quá trúng, hoặc câu đã học khá chuẩn mực nên xác suất tái hiện lại cao. Nhưng sẽ rắc rối to nếu dùng nó để viết văn, vì rất dễ sa vào đạo văn mà không biết.
Vấn đề đau đầu nữa là vì nó bung ra văn bản có xác suất cao trong tập huấn luyện, nên thi thoảng nó nói rất chi là bậy. Thế nên đã bị gắn cái mác vẹt-ngẫu-nhiên (stochastic parrot), khái niệm cu-te này đã khiến tác giả ăn đủ trong vụ làm xàm mấy năm trước ở Google.

## Để chống lại việc nói bậy có mấy cách
một là rà soát tập huấn luyện. Cái này thì tốn công tốn của. Cách khác là dạy nó nói cho hay, chẳng hạn như em xin lỗi, vợ anh nói lúc nào cũng đúng, dù chị nhà nói 1+1 = 3. Đây là một kiểu bậy khác, dù nó có tính mua vui rất cao. Cái này tốn kém, nhưng khả dĩ, và có vẻ như các cty lớn đều đang chi bộn tiền để làm. Cách nữa là truy vấn vào cơ sở dữ liệu chuẩn để kiểm chứng. Đây có lẽ là cách mà Bing, Google hay You.com đang đi.

Vì là mô hình văn bản, nên bất cứ khi nào dữ liệu diễn đạt dưới dạng văn bản, hay kỹ thuật hơn là chuỗi các token, thì máy nó học được hết. Dù dữ liệu là văn bản ngôn ngữ tự nhiên, hay mã nguồn, hình ảnh, âm thanh, video, thậm chí DNA, protein, phân tử thuốc, hồ sơ bệnh án, hay chuỗi các hành động của robot. Một khi tất thảy được kết nối, hoặc học bằng cùng một máy, khi đó khó mà bảo máy không thông minh như người.

Thực ra là chúng cóbiểu hiện như người, mà không có một mảy may ý thức việc nó đang làm, hay bất kỳ cảm xúc gì. Cuốn sách có đầy dòng lâm li bi đát hay hào khí ngất trời cũng chỉ là mực in trên giấy, tuyệt nhiên không có tí gì là thông minh cả. Sự thông minh ấy, là __do đối tượng thông minh tiếp nhận diễn ngôn ra__ (như con người) hay cảm thấy (như con mèo).

Cứ đà này, ta đang tiến tới cảnh làm-một-lần-rồi-thôi, __ít ra trong 10 năm tới__ cho đến khi có cách tiếp cận tốt hơn, mạnh mẽ hơn thay thế.
#AI, #LLM, #ChatGPT
Nguồn ảnh: Luke Skyward

![](https://scontent.fsgn13-2.fna.fbcdn.net/v/t39.30808-6/328685351_731099775396124_7885186727972056714_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=730e14&_nc_ohc=0zQIveqrPocAX8sY2b2&_nc_ht=scontent.fsgn13-2.fna&oh=00_AfCU98Pu_e7FeiHFVX1wQ6Ye3mzA5ab5XRuobz3zS9ubqg&oe=63E55552)