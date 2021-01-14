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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from sklearn import set_config
set_config(display='diagram') # Để trực quan hóa pipeline

# Để show hết dòng, cột khi hiển thị
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Để ẩn đi warnings :(
import warnings
warnings.filterwarnings('ignore')
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
data_df =pd.read_csv('../datasets/full_data.csv')
data_df.sample(n=10)
```

### Dữ liệu có bao nhiêu dòng và bao nhiêu cột?

```python
data_df.shape
```

### Mỗi dòng có ý nghĩa gì? Có vấn đề các dòng có ý nghĩa khác nhau không?


Quan sát sơ qua thì ta thấy mỗi dòng chứa thông tin của một chỗ ở. Mà cụ thể hơn, do dữ liệu được nhóm em cào nên đó là chỗ ở cho thuê cho 2 người, trong 1 ngày 1 đêm, được cào trong giai đoạn từ ngày ... đến ngày.

Nhìn sơ qua (bằng cách chạy `data_dfsample()` nhiều lần) thì thấy có vẻ có một số dòng bị vấn đề ở cột "Diện tích". Cụ thể, thay vì chứa thông tin về "x m²" thì sẽ là "Phòng tắm riêng trong phòng", "Ban công", "Điều hòa không khí",... 

Có thể đây là lỗi gặp phải trong quá trình nhóm em cào tự động, những dòng này cần phải được loại bỏ. Sau đó, sẵn tiện thì lược bỏ luôn "m2" trong cột này, chuyển sang dạng số luôn.

```python
data_df.drop(data_df[data_df['Diện tích'].str.find('m²') == -1].index,inplace=True)
data_df.shape
```

```python
data_df['Diện tích'] = data_df['Diện tích'].str[:-3]
data_df['Diện tích'] = pd.to_numeric(data_df['Diện tích'], errors='coerce')
data_df['Diện tích'].dtypes
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

```python
for col in set(data_df.columns) - set(["price", "City", "Diện tích"]):
    print(data_df[col].unique())
```

---


## Đưa ra câu hỏi cần trả lời


Với dữ liệu này, nhóm em đã xác định từ đầu cột output sẽ là "price" - Bài toán regression. Do đó, câu hỏi cần được trả lời sẽ là:

*Output - giá thuê chỗ ở (cho 2 người, trong 1 ngày 1 đêm, đơn vị: VNĐ) -* được tính từ *input - các thông tin của chỗ ở* theo công thức nào?

Việc tìm ra câu trả lời cho câu hỏi này góp phần nào tham khảo vào quá trình đưa ra quyết định cho cả 2 phía: bên cho thuê và bên thuê chỗ ở khi đi du lịch. Bên cho thuê có thể dựa vào đó để đưa ra giá tiền thuê sao cho phù hợp với thị trường, dựa trên những tiêu chí/ thông tin của tài sản. Cũng như cần chuẩn bị những tiêu chí gì để "tối ưu" giá cho thuê lên. Ngược lại, bên đi thuê sẽ dựa theo các tiêu chí mình cần để chuẩn bị ngân sách cho chỗ ở khi du lịch... 

Ở giai đoạn này, thông tin của chỗ ở - input sẽ cần được lọc bớt. Vì hiện tại dữ liệu có khá nhiều cột và nhóm cảm thấy một số cột là không cần thiết/ không ảnh hưởng tới output (ít nhất là với góc độ của thị trường Việt Nam).  

```python
drop_cols = ['Bể sục','Máy fax', 'Ổ cắm cho iPod', 'Trò chơi board game/giải đố','Không gây dị ứng','Dịch vụ báo thức', 'Lối vào riêng', 'Chăn điện', 'Sản phẩm lau rửa',  'Đầu đĩa CD',  'Đầu đĩa DVD', ' đĩa DVD và nhạc cho trẻ em', 'Thiết bị báo carbon monoxide', 'Đài radio', 'Dịch vụ báo thức']
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
X_df =data_df.drop("price", axis=1)
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


## Khám phá dữ liệu (tập huấn luyện)





### Mỗi cột input hiện đang có kiểu dữ liệu gì? Có cột nào có kiểu dữ liệu chưa phù hợp để có thể xử lý tiếp không?

```python
train_X_df.dtypes
```

Các cột đều có kiểu dữ liệu phù hợp.


### Với mỗi cột input có kiểu dữ liệu dạng số, các giá trị được phân bố như thế nào?


Chỉ có cột "Diện tích" là kiểu dạng số . 

```python
num_cols = ['Diện tích']
X_df = train_X_df[num_cols]
def missing_ratio(df):
    return (df.isna().mean() * 100).round(1)
def lower_quartile(df):
    return df.quantile(0.25).round(1)
def median(df):
    return df.quantile(0.5).round(1)
def upper_quartile(df):
    return df.quantile(0.75).round(1)
X_df.agg([missing_ratio, 'min', lower_quartile, median, upper_quartile, 'max'])
```




### Với mỗi cột input có kiểu dữ liệu không phải dạng số, các giá trị được phân bố như thế nào?

```python
cat_cols = list(set(train_X_df.columns) - set(num_cols))
X_df =train_X_df[cat_cols]
def missing_ratio(df):
    return (df.isna().mean() * 100).round(1)
def num_values(df):
    return df.nunique()
def value_ratios(c):
    return dict((c.value_counts(normalize=True) * 100).round(1))
X_df.agg([missing_ratio, num_values, value_ratios])
```

Một số trường có vẻ bị chênh lệch quá nhiều giưa 2 giá trị 0-1. Có thể kể tới như "Giường xếp", "Giấy vệ sinh",... Liệu có cần bỏ các cột này?


---


## Tiền xử lý (tập huấn luyện)

```python
train_X_df.shape
```

```python
class ColAdderDropper(BaseEstimator, TransformerMixin):
    def __init__(self, may_pha_tra_cf=True,
                bep_an=True, phong_tam=True, ket_sat=True,toilet=True,
                TV=True, dieu_hoa=True,lot_san=True, tu_quan_ao=True,
                view=True, ban_an=True, ui_quan_ao=True, giuong=True,
                tien_ich_cho_tre_em=True, ho_boi=True,
                khu_ngoai_troi=True, tien_ich_phong=True):
        
        self.may_pha_tra_cf = may_pha_tra_cf
        self.bep_an = bep_an
        self.phong_tam = phong_tam
        self.ket_sat = ket_sat
        self.toilet = toilet
        self.TV = TV
        self.dieu_hoa = dieu_hoa
        self.lot_san = lot_san
        self.tu_quan_ao = tu_quan_ao
        self.view = view
        self.ban_an = ban_an
        self.ui_quan_ao = ui_quan_ao
        self.giuong = giuong
        self.tien_ich_cho_tre_em = tien_ich_cho_tre_em
        self.ho_boi = ho_boi
        self.khu_ngoai_troi = khu_ngoai_troi
        self.tien_ich_phong = tien_ich_phong
        
    def fit(self, X_df, y=None):
        return self
    
    def transform(self, X_df, y=None):
        df = X_df.copy()
        if self.may_pha_tra_cf:
            col1 = 'Máy pha trà/cà phê'
            col2 = 'Máy pha Cà phê'
            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop(col2, axis=1, inplace=True)
            df = df.rename(columns={col1 : 'may_pha_tra_cf'})
        if self.bep_an:
            col1 = 'Bếp'
            col2 = 'Bếp nhỏ'
            col3 = 'Đồ bếp'
            col4 = 'Bếp nấu'
            col5 = 'Lò vi sóng'
            col6 = 'Máy nướng bánh mỳ'
            col7 = 'Lò nướng'
            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0 and df[col4][i] == 0 and df[col5][i] == 0 and df[col6][i] == 0 and df[col7][i] == 0:
                    pass
                else:
                    if df[col5][i] == 1:
                        if df[col6][i] == 1:
                            if df[col7][i] == 1:
                                df[col1][i] = 3
                            else:
                                df[col1][i] = 2
                        else:
                            df[col1][i] = 1
                    else:
                        df[col1][i] = 1


            df.drop([col2, col3, col4, col5, col6, col7], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'bep_an'})

        if self.phong_tam:
            col1 = 'Vòi sen'
            col2 = 'Bồn tắm'
            col3 = 'Bồn tắm hoặc Vòi sen'
            col4 = 'Bồn tắm spa'
            col5 = 'Phòng tắm riêng trong phòng'
            col6 = 'Phòng tắm phụ'
            col7 = 'Phòng tắm riêng'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0 and df[col4][i] == 0 and df[col5][i] == 0 and df[col6][i] == 0 and df[col7][i] == 0:
                    df[col1][i] = 1
                else:
                    if df[col2][i] == 1 or df[col3][i] == 1 or df[col5][i] == 1 or df[col6][i] == 1 or df[col7][i] == 1:
                        df[col1][i] = 2
                    elif df[col4][i] == 1:
                        df[col1][i] = 3
                    else:
                        df[col1][i] = 1
            df.drop([col2, col3, col4, col5, col6, col7], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'phong_tam'})
            
        if self.ket_sat:
            col1 = 'Két an toàn'
            col2 = 'Két an toàn cỡ laptop'
            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0:
                    pass
                else:
                    if df[col2][i] == 1:
                        df[col1][i] = 2

            df.drop(col2, axis=1, inplace=True)
            df = df.rename(columns={col1 : 'ket_sat'})
        if self.toilet:
            col1 = 'Nhà vệ sinh'
            col2 = 'Toilet phụ'
            col3 = 'Toilet chung'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop([col2, col3], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'toilet'})
        if self.TV:
            col1 = 'TV'
            col2 = 'TV màn hình phẳng'
            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop(col2, axis=1, inplace=True)
            df = df.rename(columns={col1 : 'TV'})
        if self.dieu_hoa:
            col1 = 'Quạt máy'
            col2 = 'Điều hòa không khí'
            col3 = 'Hệ thống sưởi'
            col4 = 'Máy điều hòa độc lập cho từng phòng'
            col5 = 'Có lắp đặt máy lọc không khí'
            col6 = 'Lò sưởi'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0 and df[col4][i] == 0 and df[col5][i] == 0 and df[col6][i] == 0:
                    pass
                else:
                    if df[col2][i] == 1 or df[col4][i] == 1:
                        df[col1][i] = 2
                    if df[col3][i] == 1 or df[col6][i] == 1:
                        df[col1][i] = 3
                    if df[col5][i] == 1:
                        df[col1][i] = 4


            df.drop([col2, col3, col4, col5, col6], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'dieu_hoa'})
        if self.lot_san:
            col1 = 'Sàn trải thảm'
            col2 = 'Sàn lát gạch/đá cẩm thạch'
            col3 = 'Sàn lát gỗ'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0:
                    pass
                else:
                    if df[col2][i] == 1:
                        df[col1][i] = 2
                    if df[col3][i] == 1:
                        df[col1][i] = 3

            df.drop([col2, col3], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'lot_san'})
        if self.tu_quan_ao:
            col1 = 'Tủ hoặc phòng để quần áo'
            col2 = 'Giá treo quần áo'
            col3 = 'Phòng thay quần áo'
            col4 = 'Giá phơi quần áo'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0 and df[col4][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop([col2, col3, col4], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'tu_quan_ao'})            
        if self.view:
            col1 = 'Nhìn ra thành phố'
            col2 = 'Nhìn ra hồ bơi'
            col3 = 'Tầm nhìn ra khung cảnh'
            col4 = 'Nhìn ra sông'
            col5 = 'Nhìn ra địa danh nổi tiếng'
            col6 = 'Nhìn ra vườn'
            col7 = 'Hướng nhìn sân trong'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0 and df[col4][i] == 0 and df[col5][i] == 0 and df[col6][i] == 0 and df[col7][i] == 0:
                    pass
                else:
                    if df[col2][i] == 1:
                        df[col1][i] = 1
                    if df[col1][i] == 1 or df[col3][i] == 1 or df[col7][i] == 1:
                        df[col1][i] = 2
                    if df[col6][i] == 1:
                        df[col1][i] = 3
                    if df[col4][i] == 1 or df[col5][i] == 1:
                        df[col1][i] = 4

            df.drop([col2, col3, col4, col5, col6, col7], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'view'})
        if self.ban_an:
            col1 = 'Bàn ăn'
            col2 = 'Khu vực phòng ăn'
            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop(col2, axis=1, inplace=True)
            df = df.rename(columns={col1 : 'ban_an'})
        if self.ui_quan_ao:
            col1 = 'Tiện nghi ủi'
            col2 = 'Bàn ủi'
            col3 = 'Bàn ủi li quần'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop([col2, col3], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'ui_quan_ao'})
        if self.giuong:
            col1 = 'Giường cực dài (> 2 mét)'
            col2 = 'Giường sofa'
            col3 = 'Giường xếp'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0:
                    df[col1][i] = 1
                else:
                    if df[col3][i] == 1:
                        df[col1][i] = 1
                    if df[col1][i] == 1 or df[col2][i] == 1:
                        df[col1][i] = 2

            df.drop([col2, col3], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'giuong'})
        if self.tien_ich_cho_tre_em:
            col1 = 'Ghế cao dành cho trẻ em'
            col2 = 'Cửa an toàn cho trẻ nhỏ'
            col3 = 'Nắp che ổ cắm điện an toàn'
            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop([col2, col3], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'tien_ich_cho_tre_em'})
        if self.ho_boi:
            col1 = 'Hồ bơi trên sân thượng'
            col2 = 'Hồ bơi có tầm nhìn'
            col3 = 'Hồ bơi riêng'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0 and df[col3][i] == 0:
                    pass
                else:
                    if df[col3][i] == 1:
                        df[col1][i] = 1
                    if df[col1][i] == 1 or df[col2][i] == 1:
                        df[col1][i] = 2

            df.drop([col2, col3], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'ho_boi'})            
        if self.khu_ngoai_troi:
            col1 = 'Khu vực ăn uống ngoài trời'
            col2 = 'Bàn ghế ngoài trời'
            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0:
                    pass
                else:
                    df[col1][i] = 1

            df.drop(col2, axis=1, inplace=True)
            df = df.rename(columns={col1 : 'khu_ngoai_troi'})
        if self.tien_ich_phong:
            col1 = 'Áo choàng tắm'
            col2 = 'Khăn tắm'
            col3 = 'Khăn tắm/Bộ khăn trải giường (có thu phí)'

            for i in df.index:
                if df[col1][i] == 0 and df[col2][i] == 0:
                    pass
                if df[col3][i] == 1:
                    df[col1][i] = 0
                if df[col1][i] == 1 or df[col2][i] == 1:
                    df[col1][i] = 1

            df.drop([col2, col3], axis=1, inplace=True)
            df = df.rename(columns={col1 : 'tien_ich_phong'})
        return df
```

```python
nume_cols = ['Diện tích']
unorder_cate_cols = ['City']
order_cate_cols = ['Đồ vệ sinh cá nhân miễn phí',
       'tien_ich_phong', 'ket_sat', 'toilet', 'Bàn làm việc',
       'Khu vực tiếp khách', 'TV', 'Dép', 'Tủ lạnh', 'Điện thoại',
       'Máy sấy tóc', 'lot_san', 'Ấm đun nước điện', 'Truyền hình cáp',
       'tu_quan_ao', 'view',
       'Hệ thống cách âm', 'Minibar', 'WiFi miễn phí', 'Ghế sofa', 'bep_an',
       'Máy giặt', 'Đồng hồ báo thức', 'ban_an',
       'Ổ điện gần giường', 'Giấy vệ sinh',
        'ui_quan_ao', 'giuong',
       'Truyền hình vệ tinh', 'Ban công', 'Quyền sử dụng Executive Lounge',
       'Ra trải giường', 'may_pha_tra_cf', 'Máy sấy quần áo',
       'Có phòng thông nhau qua cửa nối', 'Chậu rửa vệ sinh (bidet)',
       'Các tầng trên đi lên bằng thang máy',
       'Xe lăn có thể đi đến mọi nơi trong toàn bộ khuôn viên', 'Máy vi tính',
       'dieu_hoa', 'tien_ich_cho_tre_em', 'Hoàn toàn nằm ở tầng trệt',
       'Các tầng trên chỉ lên được bằng cầu thang', 'Nước rửa tay', 'ho_boi',
       'Sách', 'khu_ngoai_troi', 'Phòng xông hơi', 'Sân trong']
# 
nume_col_transformer = SimpleImputer(strategy = 'mean')
unorder_cate_col_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('one_hot_encoder', OneHotEncoder())])
order_cate_col_transformer = SimpleImputer(strategy = 'most_frequent')
col_transformer = ColumnTransformer(transformers = [
    ('nume_col_transformer', nume_col_transformer, nume_cols),
    ('unorder_cate_col_transformer', unorder_cate_col_transformer, unorder_cate_cols),
    ('order_cate_col_transformer', order_cate_col_transformer, order_cate_cols)])

# 
preprocess_pipeline = Pipeline(steps = [('col_adderdropper', ColAdderDropper()),
                                           ('col_transformer', col_transformer),
                                           ('col_normalizer', StandardScaler())])
```

```python
preprocessed_train_X = preprocess_pipeline.fit_transform(train_X_df)
```

```python
preprocess_pipeline
```

---


## Tiền xử lý (tập validation)

```python
preprocessed_val_X = preprocess_pipeline.transform(val_X_df)
```

---


## Tiền xử lý + mô hình hóa


### Tìm mô hình tốt nhất


### Đánh giá mô hình tìm được
