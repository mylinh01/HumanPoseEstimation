# XÂY DỰNG ỨNG DỤNG ƯỚC TÍNH TƯ THẾ CON NGƯỜI THEO THỜI GIAN THỰC SỬ DỤNG MÔ HÌNH MOVENET
## 1. Giới thiệu 
### 1.1. Giới thiệu đề tài
Phát hiện đối tượng là một lĩnh vực quan trọng trong thị giác máy tính và có ứng dụng rộng rãi trong nhiều lĩnh vực "thực tế". Trong đó, việc trích xuất các điểm chính được sử dụng để ước tính tư thế là một nhiệm vụ quan trọng. Các điểm chính có thể là các thành phần khác nhau như các phần của khuôn mặt hoặc các phần của cơ thể. Việc ước tính tư thế đặc biệt là một phần quan trọng của phát hiện điểm chính, trong đó các điểm đại diện cho các phần của cơ thể con người.

Công nghệ phát hiện và theo dõi tư thế thời gian thực với hiệu suất cao đang đẩy mạnh những xu hướng chính trong lĩnh vực thị giác máy tính. Ví dụ, việc theo dõi tư thế của con người trong thời gian thực cho phép máy tính có được sự hiểu biết chi tiết và tự nhiên hơn về hành vi con người. Điều này có tác động to lớn đến nhiều lĩnh vực khác nhau, chẳng hạn như xe tự lái.

### 1.2. Mục tiêu 
Mục tiêu là nhận tư thế của nhiều người trong khung hình theo thời gian thực bằng cách sử dụng bản đồ nhiệt để khoanh vùng chính xác các điểm chính của con người. 
### 1.3. Kết quả
Kết quả dự kiến của dự án là có khả năng ước tính tư thế của nhiều người trong thời gian thực trên các hình ảnh, video và từ webcam. Mô hình sẽ được triển khai thành một ứng dụng web, cho phép người dùng tương tác và theo dõi tư thế của họ hoặc của người khác trong thời gian thực.

## 2. Tổng quan về kiến thức cơ sở
### 2.1.	Ước tính tư thế con người (Human Pose Estimation)
Ước tính tư thế con người (Human Pose Estimation) là một cách xác định và phân loại các khớp trong cơ thể con người. Về cơ bản, đó là một cách để nắm bắt một tập hợp tọa độ cho từng khớp (cánh tay, đầu, thân, v.v.), được gọi là điểm chính có thể mô tả tư thế của một người. Kết nối giữa các điểm này được gọi là một cặp.
Kết nối được hình thành giữa các điểm phải có ý nghĩa, điều đó có nghĩa là không phải tất cả các điểm đều có thể tạo thành một cặp. Ngay từ đầu, mục đích của ước tính tư thế con người là tạo ra một biểu tượng giống như khung xương của cơ thể con người và sau đó xử lý thêm cho các ứng dụng dành riêng cho nhiệm vụ.
### 2.2.	Tổng quan về MoveNet
#### 2.2.1.	MoveNet
MoveNet là một mô hình mạng thần kinh tích chập chạy trên hình ảnh RGB và dự đoán vị trí khớp của con người trong khung hình ảnh. Mô hình này được cung cấp trên TF Hub với hai biến thể, được gọi là Lightning và Thunder. Lightning dành cho các ứng dụng quan trọng về độ trễ, trong khi Thunder dành cho các ứng dụng yêu cầu độ chính xác cao. Cả hai mô hình này được thiết kế để chạy trong trình duyệt sử dụng TensorFlow.js hoặc trên các thiết bị sử dụng TensorFlow Lite, nhắm mục tiêu các hoạt động vận động/thể dục, điều này chứng tỏ rất quan trọng đối với các ứng dụng thể dục trực tiếp, sức khỏe và sức khỏe.

#### 2.2.2.	Kiến trúc MoveNet
MoveNet là một mô hình ước tính từ dưới lên, sử dụng bản đồ nhiệt để định vị chính xác các điểm chính của con người. Kiến trúc bao gồm hai thành phần: bộ trích xuất đặc trưng và bộ đầu dự đoán. Sơ đồ dự đoán tuân theo CenterNet một cách lỏng lẻo, với những thay đổi đáng chú ý giúp cải thiện cả tốc độ và độ chính xác. Tất cả các mô hình đều được đào tạo bằng cách sử dụng API phát hiện đối tượng TensorFlow.

Trình trích xuất tính năng trong MoveNet là MobileNetV2 với mạng kim tự tháp tính năng (FPN) được đính kèm, cho phép tạo ra bản đồ tính năng có độ phân giải cao. Có bốn đầu dự đoán được gắn vào trình trích xuất đặc trưng, chịu trách nhiệm dự đoán:

-	Person center heatmap: dự đoán trung tâm hình học của các phiên bản người
-	Keypoint regression field: dự đoán tập hợp đầy đủ các điểm chính cho một người, được sử dụng để nhóm các điểm chính thành các phiên bản
-	Person keypoint heatmap: dự đoán vị trí của tất cả các điểm chính, không phụ thuộc vào phiên bản người
-	2D per-keypoint offset field: dự đoán độ lệch cục bộ từ từng pixel bản đồ tính năng đầu ra đến vị trí pixel phụ chính xác của từng điểm chính

![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/c0737afa-c648-4703-b3fa-fb5469f06391)

Hình 1.1. Kiến trúc MoveNet

Mặc dù những dự đoán này được tính toán song song, nhưng người ta có thể hiểu rõ hơn về hoạt động của mô hình bằng cách xem xét chuỗi hoạt động sau:

Bước 1: Bản đồ nhiệt trung tâm người được sử dụng để xác định trung tâm của tất cả các cá nhân trong khung, được định nghĩa là trung bình cộng của tất cả các điểm chính thuộc về một người. Vị trí có điểm số cao nhất (có trọng số theo khoảng cách nghịch đảo từ trung tâm khung) được chọn.

Bước 2: Tập hợp các điểm chính ban đầu cho người được tạo bằng cách cắt đầu ra hồi quy điểm chính từ pixel tương ứng với trung tâm đối tượng. Vì đây là một dự đoán trung tâm – phải hoạt động trên các quy mô khác nhau – nên chất lượng của các điểm chính hồi quy sẽ không chính xác lắm.
Bước 3: Mỗi pixel trong bản đồ nhiệt của điểm chính được nhân với trọng số tỷ lệ nghịch với khoảng cách từ điểm chính hồi quy tương ứng. Điều này đảm bảo rằng chúng tôi không chấp nhận các điểm chính từ những người trong nền, vì chúng thường sẽ không ở gần các điểm chính bị hồi quy và do đó sẽ có điểm số thấp.

Bước 4: Tập hợp dự đoán điểm chính cuối cùng được chọn bằng cách truy xuất tọa độ của các giá trị bản đồ nhiệt tối đa trong mỗi kênh điểm chính. Sau đó, các dự đoán độ lệch 2D cục bộ được thêm vào các tọa độ này để đưa ra các ước tính tinh chỉnh. Xem hình bên dưới minh họa bốn bước này.
 ## 3. Xây dựng ứng dụng
 ### 3.1. Xây dựng ứng dụng
#### Tải mô hình từ TF hub
![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/728f039c-5194-4c3e-bf7e-fae3e075e762)

Sử dụng mô hình MoveNet.MultiPose để có thể phát hiện nhiều người trong khung hình cùng một lúc trong khi vẫn đạt được tốc độ thời gian thực. Biến thể “Lightning” có thể chạy ở tốc độ >30FPS trên hầu hết các máy tính xách tay hiện đại và phát hiện đồng thời tối đa 6 người trong khi đạt được hiệu suất tốt.

Input:
Khung video hoặc hình ảnh, được biểu thị dưới dạng tensor int32 có hình dạng động: 1xHxWx3, trong đó H và W cần phải là bội số của 32 và kích thước lớn hơn được khuyến nghị là 256. 

Output:
Một tensor float32 có dạng [1, 6, 56].
-	Kích thước đầu tiên luôn bằng 1.
-	Kích thước thứ hai tương ứng với số lần phát hiện phiên bản tối đa. Mô hình có thể phát hiện đồng thời tối đa 6 người trong khung hình.
-	Kích thước thứ ba đại diện cho các vị trí và điểm số của hộp giới hạn/điểm chính được dự đoán. 17 * 3 phần tử đầu tiên là vị trí điểm chính và điểm số ở định dạng: [y_0, x_0, s_0, y_1, x_1, s_1, …, y_16, x_16, s_16], trong đó y_i, x_i, s_i là tọa độ yx (được chuẩn hóa thành khung hình ảnh, ví dụ: phạm vi trong [0,0, 1,0]) và điểm tin cậy của khớp thứ i tương ứng. Thứ tự của 17 điểm mấu chốt là: [mũi, mắt trái, mắt phải, tai trái, tai phải, vai trái, vai phải, khuỷu tay trái, khuỷu tay phải, cổ tay trái, cổ tay phải, hông trái, hông phải, đầu gối trái, đầu gối phải, mắt cá chân trái, mắt cá chân phải]. 5 phần tử còn lại [ymin, xmin, ymax, xmax, score] đại diện cho vùng của hộp giới hạn (theo tọa độ chuẩn hóa) và điểm tin cậy của thể hiện.
#### Vẽ các keypoint
-	Lấy các thông số (hình dạng): chiều rộng, chiều cao
-	Không chuẩn hóa tọa độ đầu ra bằng cách thay đổi các điểm chính với các tham số
-	Lặp lại các điểm chính không chuẩn hóa và vẽ các vòng tròn có điểm tin cậy cao hơn ngưỡng đặt trước
![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/ba625720-7e91-4ea6-8f8d-cad923dc99ca)

#### Vẽ kết nối các keypoint
-	Lặp qua từng cạnh và vẽ các cạnh
-	Nhận các điểm cạnh và giá trị chính tả được liên kết
-	Vẽ các đường có điểm tin cậy cao hơn ngưỡng đặt trước
![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/6cd20553-d282-4ce5-a5b1-4cecfac36036)

#### Lặp lại các bước cho từng đối tượng
![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/10244f5f-1424-4130-be51-22f94789c845)
### 3.2. Kết quả
Kết quả trên local
![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/e036d27e-1581-4419-8b95-cc5782d154b1)

Sử dụng streamlit để xây dựng web app ước tính tư thế nhiều người theo thời gian thực

Input có thể chọn là Video để load video hoặc Webcam

![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/05bea538-3678-4f4f-bcea-d241a38a5f22)


Output là video ước tính tư thế nhiều người theo thời gian thực
![image](https://github.com/mylinh01/HumanPoseEstimation/assets/91240116/7c4759bf-f4aa-46c9-bb57-e06b138aac31)




