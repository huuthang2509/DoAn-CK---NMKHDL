---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# ĐỒ ÁN CUỐI KỲ (NMKHDL - CQ2018/2)
### Giảng viên hướng dẫn: Thầy Trần Trung Kiên
### Trợ giảng: Thầy Hồ Xuân Trường

__Thông tin nhóm__: 

* STT: 46

* Họ và tên sinh viên: Nguyễn Hữu Huân - MSSV: 1712466

* Họ và tên sinh viên: Đặng Hữu Thắng - MSSV: 18120555


---


## Import

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

# for model picking
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
```

----


## Thu thập dữ liệu


Dữ liệu mà nhóm em dùng cho đồ án cuối kỳ được tự thu thập từ trang [Booking.com](https://booking.com) - Trang web về du lịch cung cấp dịch vụ đặt chỗ, đặt phòng khách sạn ở trực tuyến trên toàn thế giới. Dữ liệu nhóm em thu thập chứa thông tin về chỗ ở (bao gồm cả khách sạn, nhà nghỉ, homestay,..) ở 4 địa điểm du lịch của Việt Nam: Hà Nội, Đà Nẵng, Đà Lạt, Thành phố Hồ Chí Minh.

Dữ liệu trên trang web này được phép thu thập, được mô tả cụ thể thông qua file robots.txt.

Phần source code cho quá trình cào dữ liệu về  được lưu ở file notebook [crawling_data.ipynb](https://github.com/ref-to-uploaded-notebook).

Toàn bộ dữ liệu thô sau khi cào về lưu ở file: "full_data.csv".


---


## Khám phá dữ liệu (đủ để xác định câu hỏi) + tiền xử lý 


Do file "full_data.csv" là tất cả dữ liệu mà nhóm em tự cào được, nên nhóm sẽ cần khám phá một ít trên dữ liệu để đưa ra câu hỏi cần trả lời, sau đó tiến hành kiểm tra, làm sạch một vài vấn đề cơ bản (vì tất cả hoàn toàn là dữ liệu thô) và ngay lập tức tách tập validation và test ra khỏi dữ liệu. Như đã được thầy đề cập ở BT03, điều này để tránh hiểu quá sâu dữ liệu, làm mất đi tính khách quan khi đánh giá kết quả.

```python
data_df = pd.read_csv('../datasets/full_data.csv')
data_df.sample(n=10)
```

### Dữ liệu có bao nhiêu dòng và bao nhiêu cột?

```python
data_df.shape
```

### Mỗi dòng có ý nghĩa gì? Có vấn đề các dòng có ý nghĩa khác nhau không?


Quan sát sơ qua thì ta thấy mỗi dòng chứa thông tin của một chỗ ở. Mà cụ thể hơn, do dữ liệu được nhóm em cào nên đó là chỗ ở cho thuê cho 2 người, trong 1 ngày 1 đêm, được cào trong giai đoạn từ ngày ... đến ngày.

Nhìn sơ qua (bằng cách chạy `data_df.sample()` nhiều lần) thì thấy có vẻ có một số dòng bị vấn đề ở cột "Diện tích". Cụ thể, thay vì chứa thông tin về "x m²" thì sẽ là "Phòng tắm riêng trong phòng", "Ban công", "Điều hòa không khí",... 

Có thể đây là lỗi gặp phải trong quá trình nhóm em cào tự động, những dòng này cần phải được loại bỏ.

```python
data_df.drop(data_df[data_df['Diện tích'].str.find('m²') == -1].index,inplace=True)
```

```python
data_df.shape
```

### Dữ liệu có các dòng bị lặp không?

```python
data_df.index.duplicated().sum()
```

### Mỗi cột có ý nghĩa gì?


Vì hầu hết thông tin trên booking.com đều đã được dịch sang Tiếng Việt nên các cột nhìn vào có thể hiểu được nó chứa thông tin về gì. Ngoài thông tin về "price", "City", "Diện tích" thì các thông tin còn lại của booking đều hiển thị dưới dạng: 1 - có, 0 - không. Nên dữ liệu ở các cột còn lại đều là thuộc loại binary.

```python
list(data_df.columns)
```

```python jupyter={"outputs_hidden": true}
for col in set(data_df.columns) - set(["price", "City", "Diện tích"]):
    print(data_df[col].unique())
```

---


## Đưa ra câu hỏi cần trả lời


Với dữ liệu này, nhóm em đã xác định từ đầu cột output sẽ là "price" - Bài toán regression. Do đó, câu hỏi cần được trả lời sẽ là:

*Output - giá thuê chỗ ở (cho 2 người, trong 1 ngày 1 đêm, đơn vị: VNĐ) -* được tính từ *input - các thông tin của chỗ ở* theo công thức nào?

Việc tìm ra câu trả lời cho câu hỏi này góp phần nào tham khảo vào quá trình đưa ra quyết định cho cả 2 phía: bên cho thuê và bên thuê chỗ du lịch. Bên cho thuê có thể dựa vào đó để đưa ra giá tiền thuê sao cho phù hợp với thị trường, dựa trên những tiêu chí/ thông tin của tài sản. Cũng như cần chuẩn bị những tiêu chí gì để "tối ưu" giá cho thuê lên. Ngược lại, bên đi thuê sẽ dựa theo các tiêu chí mình cần để chuẩn bị ngân sách cho chỗ ở khi du lịch... 

Ở giai đoạn này, thông tin của chỗ ở - input sẽ cần được lọc bớt. Vì hiện tại dữ liệu có khá nhiều cột và nhóm cảm thấy một số cột là không cần thiết/ không ảnh hưởng tới output (ít nhất là với góc độ của thị trường Việt Nam).  

```python
drop_cols = ['Không gây dị ứng','Dịch vụ báo thức', 'Lối vào riêng', 'Chăn điện', 'Sản phẩm lau rửa',  'Đầu đĩa CD',  'Đầu đĩa DVD', ' đĩa DVD và nhạc cho trẻ em', 'Thiết bị báo carbon monoxide', 'Đài radio', 'Dịch vụ báo thức']
data_df.drop(drop_cols, axis=1, inplace=True)
data_df.shape
```

---


## Khám phá dữ liệu (để biết tách các tập) + tiền xử lý


Ở bước này, dữ liệu ở cột output cần được khám phá một ít để phục vụ cho việc tách tập:



* Cột output có kiểu dữ liệu gì? 

```python
# Cột output có kiểu dữ liệu gì?
data_df['price'].dtype
```

Là kiểu object - string. Cột này cần được tiền xử lý, chuyển sang dạng số vì là bài toán regression. Ngoài ra cần bỏ luôn "m²" và những dấu "," trong đó.

```python
data_df['price'] = data_df['price'].str.replace('.', '')
data_df['price'] = pd.to_numeric(data_df['price'],errors='coerce')
data_df['price'].dtype
```

* Cột này có giá trị thiếu không?

```python
data_df['price'].isna().sum()
```

---


## Tiền xử lý (tách các tập)


Sau khi đã khám phá và tiền xử lý 1 ít với dữ liệu, thì bây giờ là lúc tách tập test và validation ra khỏi dữ liệu.

```python
# Tách X và y
y_sr = data_df["price"] # sr là viết tắt của series
X_df = data_df.drop("price", axis=1)
```

```python
# Tách tập huấn luyện và tập test theo tỉ lệ 80%:20%
train_X_df, test_X_df, train_y_sr, test_y_sr = train_test_split(X_df, y_sr, test_size=0.2, 
                                                                  random_state=0)
# Tiếp tục từ tập huấn luyện tách ra tập huấn luyện và tập validation 80%:20%
train_X_df, val_X_df, train_y_sr, val_y_sr = train_test_split(X_df, y_sr, test_size=0.2, 
                                                                  random_state=0)
                                                            
```

```python
train_X_df.shape, train_y_sr.shape
```

```python
 val_X_df.shape, val_y_sr.shape
```

```python
test_X_df.shape, test_y_sr.shape
```

---


## Khám phá đữ liệu (tập huấn luyện)


---


## Tiền xử lý (tập huấn luyện)


---


## Tiền xử lý (tập validation)


---


## Tiền xử lý + mô hình hóa
