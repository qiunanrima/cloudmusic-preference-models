# CloudMusic Preference Models

基于 MS-CLAP 音频表示的个人音乐口味建模实验：加权 k-NN、播放强度回归、潜空间扩散，以及连续自回归（AR）密度比排序。

**本仓库仅包含源代码。** 不包含训练音频、喜欢歌单、播放记录、标签数据、向量缓存、个人排行榜、训练日志、模型权重或上游 MS-CLAP 权重。运行前需要提供自己的合法音频和数据。

## 相对 AR

个人模型与背景模型分别学习音频向量分布，排序分数为：

`score(x) = log p_personal(x) - log p_background(x)`

原生模式保留 1024 维坐标，不使用 PCA；只按训练集统计量逐坐标中心化、缩放。1024 维拆成 64 个 16 维块，使用带因果掩码的两层 Transformer（宽度128、4头注意力）和4分量对角高斯混合输出头。训练保存最佳验证权重、逐轮 loss、学习率及原始日志。

未收藏、低播放和未知播放次数均不是明确负例。背景数据是未标记样本，不等于“不喜欢”。百分位和密度比都不是校准后的喜欢概率。

## 环境

Python 3.11、PyTorch；音频预处理需要 PATH 中可用的 ffmpeg。先按自己的显卡安装合适的 PyTorch，再安装 `requirements.txt`。MS-CLAP 使用上游 `microsoft/msclap` 的 2023 音频塔，权重需要自行按上游许可获取并缓存。代码的离线编码器不会自动下载权重。

```powershell
python -m pip install -r requirements.txt
cd genre_model
$env:PYTHONIOENCODING='utf-8'
```

## 私有数据布局

默认项目根目录是 `genre_model/` 的上级，也可设置 `MUSIC_ROOT`。以下内容均应留在本地，不要提交：

```text
dataset_v3/labels_liked_files_20261029.json   # 自己的喜欢歌曲元数据
ncm-playback-dl/tmp/liked-files.txt           # 喜欢歌曲文件名，一行一个
ncm-playback-dl/tmp/signals_final.json        # 自己的歌单/播放记录
genre_model/cache/                          # 自己的向量缓存
models_archive/                             # 自己训练的模型与结果
```

标签 JSON 为对象数组，基本字段包括 `文件`、`歌名`、`歌手`；可提供 `风格`、`语种`、`BPM`、`时长秒` 等。缺失字段保留缺失，不用歌名猜标签。行为文件的读取及保守关联规则见 `preference_data.py`。

行为权重：`w = 0.5 + 是否在喜欢的音乐歌单 + ln(1 + 已知播放次数²)`。未知次数不增加对数项。Top200实验仅用权重筛选前200首，入选后均匀采样。

缓存应包含 `paths`、`vectors`、`model_id`、`preprocess_id`。`model_id` 为 `msclap:2023`，原生向量为1024维。`MSCLAP_CACHE`、`MSCLAP_PREPROCESS_ID` 与 `MSCLAP_VECTOR_DIM` 必须与实际缓存一致。

## 训练入口

```powershell
python preference_knn.py train --cache <自己的缓存.npz> --out <新的KNN目录>
python train_replay_model.py --model <KNN目录> --out <新的MLP目录>
python train_random_forest.py --model <KNN目录> --out <新的RF目录>
python train_conditional_diffusion.py --model <KNN目录> --out <新的个人扩散目录> --epochs 360 --patience 360 --batch-size 256
python evaluate_diffusion_background.py --personal <个人扩散目录> --out <新的背景扩散目录>
python train_ar_preference.py --split-experiment <背景扩散目录> --out <新的AR目录> --no-projection --dimension 1024 --chunks 64 --batch-size 256
```

新实验必须使用新输出目录。训练任务应串行执行，batch size 需要按显存调整。AR会复用背景扩散的个人/背景划分；没有自己的数据时无法直接训练。

### 权重前200首相对 AR

```powershell
python train_ar_top200.py --knn-model <自己的KNN目录> --background-ar <自己的完整AR目录> --cache <原生1024维缓存.npz> --out <新的Top200实验目录>
```

该脚本选取自己喜欢清单中权重最高的200首，按歌手/录音分组，并仅按样本数量平衡训练、验证和测试比例。个人与背景训练/验证数量匹配；其余已知喜欢曲不会被当作背景或负例。清单、权重、模型、曲线和排行榜保存在指定的本地输出目录。

## 音频与维度实验

`pipeline_16k_loudness_256.py` 保留了完整实验流水线：16kHz单声道、两遍 loudnorm（目标−16 LUFS、−1.5 dBTP、LRA11）、中间7秒、重采样到44.1kHz适配编码器接口，再提取原生1024维特征。其256维PCA分支只在训练歌曲上拟合。

该流水线保留本项目的实验目录约定，需要自行提供清单、既有缓存和划分文件；它不是开箱即用的数据下载器。16k有效带宽在重采样后不会恢复。响度目标也不代表每首实测值精确等于目标。

其他脚本保留研究用途；模型选择只应使用训练/验证数据。测试集用于报告，跨实验比较应使用相同留出歌曲。上游模型及依赖遵循各自许可。

## 传统机器学习实验

`traditional_preference_experiment.py` 使用冻结的音频向量，对比加权 KNN、相对 KNN、加权核密度、相对核密度、多兴趣 KMeans、正则化高斯混合、单类 SVM 和孤立森林。随机森林单独预测已知播放强度，不把未收藏歌曲标成不喜欢。相对核方法比较个人收藏支持度与背景库支持度。

`compare_kernel_dimensions.py` 展示 PCA64/128/192/256/512 与原生1024维的 KNN/KDE 结果，包括每个维数的验证选参、固定参数对照及完整排名。维数和超参数按验证集选择，测试集只用于报告；这些是正样本与未标记样本的检索代理指标。

```powershell
python -m unittest test_traditional_preference -v
python traditional_preference_experiment.py
python compare_kernel_dimensions.py
```

这些研究入口复用本项目约定的本地缓存、训练划分、元数据和实验目录。运行前应查看脚本顶部路径及依赖的本地实验产物，改为自己的数据；仓库不附带这些文件。它们不需要训练新的神经网络，但现有 CLAP 特征仍来自预训练音频编码器。传统方法保存搜索记录、指标和比较图，不伪造梯度学习率曲线。

## 标签条件与权重消融

`train_tagged_ar_experiment.py` / `run_tagged_ar_dimensions.py` 对比纯音频、只加行为权重、只加标签条件、标签与权重同时加入的相对 AR。`run_ar_pca_dimension_sweep.py` 提供纯音频维数基线。

`train_tagged_diffusion_sweep.py` 将相同条件与权重实验用于相对扩散，保存模型、原始日志、逐轮 loss/LR、完整排名与汇总图。评分使用个人/背景模型共享的固定高斯噪声；`verify_tagged_diffusion_sweep.py` 从导出结果复核划分、曲库覆盖和 AUC。

标签估计器只读取训练集注释，为所有歌曲统一生成条件，训练歌曲排除自身；不把待测歌曲是否有标签直接作为收藏身份线索。缺失标签较多时，这种估计条件可能带来噪声，不能把条件模型效果等同于真实标签的价值。行为权重用于归一化加权损失，不同时重复做加权采样。

```powershell
python -m unittest test_tagged_ar test_tagged_diffusion -v
python run_ar_pca_dimension_sweep.py --dimensions 64 128 192 256 512
python run_tagged_ar_dimensions.py
python train_tagged_diffusion_sweep.py
```

所有训练串行执行。默认实验路径用于复现实验流程，需要自行准备；个人音频、标签、向量、模型、日志与排行榜继续由 `.gitignore` 排除。
