# BIOCLIP: A Vision Foundation Model for the Tree of Life
使用Mindspore对BioCLIP模型进行复现。[Mindspore官网](https://www.mindspore.cn/)，[BioCLIP论文](https://imageomics.github.io/bioclip/)  

## 数据集和预训练模型
数据集、预训练模型和BioCLIP模型的下载链接：[Dataset, Model](https://bhpan.buaa.edu.cn/link/AA0EC2E1AA4B9C4FCEBF5551E8C1245AB7)，提取码：wRN3

## 代码运行
所有Python代码均在`src`目录下，运行前请确保已安装Mindspore和相关依赖。  
执行以下命令以运行代码，更多参数参考`params.py`：  
```bash
python -m src.evaluation.zero_shot --gpu 0 --data_root /path_to_data --logs /path_to_logs --pretrained /path_to_model_ckpt/BIOCLIP.ckpt
```
同时，也提供了jupyter notebook文件`main.ipynb`，可以直接在Jupyter环境中运行。  

## 评估结果
Top-1准确率评估结果：  
| 模型 | PlantNet | Insects |
| ---- | -------- | ------- |
| OpenCLIP | 62.70 | 9.21 |
| BioCLIP | 94.80 | 33.89 |

Top-3准确率评估结果：  
| 模型 | PlantNet | Insects |
| ---- | -------- | ------- |
| OpenCLIP | 84.30 | 18.03 |
| BioCLIP | 99.10 | 55.77 |

Top-5准确率评估结果：  
| 模型 | PlantNet | Insects |
| ---- | -------- | ------- |
| OpenCLIP | 89.40 | 22.64 |
| BioCLIP | 99.50 | 65.14 |

