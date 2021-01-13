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


Dữ liệu mà tụi em dùng cho đồ án cuối kỳ được tự thu thập từ trang [Booking.com](https://booking.com) - Trang web về du lịch cung cấp dịch vụ đặt chỗ, đặt phòng khách sạn ở trực tuyến trên toàn thế giới. Dữ liệu tụi em thu thập chứa thông tin về chỗ ở (bao gồm cả khách sạn, nhà nghỉ, homestay,..) ở 4 địa điểm du lịch của Việt Nam: Hà Nội, Đà Nẵng, Đà Lạt, Thành phố Hồ Chí Minh.

Dữ liệu trên trang web này được phép thu thập thông qua file robots.txt.


---


## Khám phá dữ liệu (để xác định câu hỏi)


---


## Đưa ra câu hỏi cần trả lời


---


## Khám phá dữ liệu (để biết tách các tập)


---


## Tiền xử lý (tách các tập)


---


## Khám phá đữ liệu (tập huấn luyện)


---


## Tiền xử lý (tập huấn luyện)


---


## Tiền xử lý (tập validation)


---


## Tiền xử lý + mô hình hóa
