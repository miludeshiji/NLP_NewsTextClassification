# 天池NPL-新闻文本分类
## 【AI入门系列】NLP新闻分类学习赛
* [比赛链接](https://tianchi.aliyun.com/competition/entrance/531810)
* 结果：0.9386
---
## 运行方法
1. 到天池[赛题与数据](https://tianchi.aliyun.com/competition/entrance/531810/information)下载数据集并复制到text_data文件夹下
2. 运行`py -3.8 -m venv .venv`创建python虚拟环境（请确保你有python-3.8版本）。
3. 运行`.\.venv\Scripts\activate `进入虚拟环境。
4. 运行`pip install -r requirements.txt`下载环境配置
 **注意**:此项目的cuda版本为11.8，所以请确保你的cuda也是11.8版本，如果是的话将requirements.txt中的注释取消后再下载环境配置，否则请至pytorch官网下载对应版本的torch。
5. 在虚拟环境中运行`py .\main.py`即可开始训练。

---
## 文件夹结构
```
|-- README.md                       # 介绍代码框架及解题思路
|-- text_data                       # 数据集
    |-- test_a.csv                  # 测试数据集
    |-- test_a_sample_submit.csv
    |-- train_set.csv
|-- model                           # 模型文件夹
|-- processed_data                  # 存放预处理后的数据集
|-- result                          # 存放预测结果
|-- feature.py                      # 对数据进行预处理
|-- model.py                        # 训练模型
|-- main.py                         # 源代码
|-- requirements.txt                # 环境配置文件

```