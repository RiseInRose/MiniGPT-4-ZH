MiniGPT-4: 使用先进的大型语言模型增强视觉语言理解
作者为朱德尧、陈俊、沈晓倩、李翔和Mohamed Elhoseiny。*表示贡献相等。

所属机构为沙特阿拉伯国王科技大学。

在线演示请点击图像与MiniGPT-4交互。
[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

更多示例请参阅项目主页。


如果翻译对您有帮助，请帮忙右上角 点击 star.  
[欢迎加入国内AI商业应用交流群](#国内交流群)

---

介绍：
MiniGPT-4使用一个投影层将来自BLIP-2的冻结视觉编码器与冻结LLM Vicuna对齐。我们使用两个阶段进行训练。第一个阶段使用大约500万对图片和文本的对齐数据，在4个A100上进行的传统预训练阶段，耗时10小时。在第一个阶段之后，Vicuna能够理解图像，但生成能力受到严重影响。
为了解决这个问题并提高可用性，我们提出了一种新颖的方法，通过模型本身和ChatGPT一起创建高质量的图像文本对。基于此，我们创建了一个小规模（总共3500对）但高质量的数据集。
在对话模板中，使用这个数据集进行第二次微调，以显著提高其生成可靠性和整体可用性。令人惊讶的是，这个阶段的计算效率很高，只需一个A100大约7分钟。
MiniGPT-4展示了许多类似于GPT-4中展示的新兴视觉语言能力。


入门指南：
### 安装

**1.准备代码和环境**

请先将我们的代码库克隆到本地，创建一个Python环境，然后通过以下命令激活它

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2.准备预训练的Vicuna权重**

当前版本的MiniGPT-4是建立在Vicuna-13B v0版本之上的。请参考我们的说明[here](PrepareVicuna.md)来准备Vicuna权重。

### here 的翻译如下：
如何准备Vicuna权重

Vicuna是一种基于LLAMA的LLM，性能接近于ChatGPT，并且是开源的。我们当前使用的是Vicuna-13B v0版本。

为了准备Vicuna的权重，首先从 https://huggingface.co/lmsys/vicuna-13b-delta-v0  下载Vicuna的增量权重。如果你已经安装了git-lfs（https://git-lfs.com），可以通过以下方式完成：

```bash
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0
```
请注意，这并不是直接可用的工作权重，而是工作权重与LLAMA-13B原始权重之间的差异（由于LLAMA的规则，我们无法分发LLAMA的权重）。

然后，您需要获取原始的LLAMA-13B权重，可以按照HuggingFace提供的说明[here](https://huggingface.co/docs/transformers/main/model_doc/llama) 或者从互联网上下载。

### 原始权重获取如下：
提示：

可以通过填写表格来获取LLaMA模型的权重。当然，热心的“网友”已经帮忙泄漏出来了。
下载地址：https://github.com/facebookresearch/llama/issues/149 里面有很多选择，不安装的话，可以点击链接，使用迅雷下载。

```
You can download normally, or use these commands from the Kubo CLI:

# Optional: Preload the 7B model. Retrieves the content you don't have yet. Replace with another CID, as needed.
ipfs refs -r QmbvdJ7KgvZiyaqHw5QtQxRtUd7pCAdkWWbzuvyKusLGTw

# Optional: Pin the 7B model. The GC removes old content you don't use, this prevents the model from being GC'd if enabled.
ipfs pin add QmbvdJ7KgvZiyaqHw5QtQxRtUd7pCAdkWWbzuvyKusLGTw

# Download from IPFS and save to disk via CLI:
ipfs get QmbvdJ7KgvZiyaqHw5QtQxRtUd7pCAdkWWbzuvyKusLGTw --output ./7B


```


下载完权重之后，需要使用转换脚本将它们转换为Hugging Face Transformers格式。可以使用以下命令（示例）调用脚本：
脚本地址：https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

转换完成后，可以通过以下方式加载模型和分词器：

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

当这两个权重准备好后，我们可以使用Vicuna团队的工具来创建真正的工作权重。首先，安装与v0 Vicuna兼容的库：

```bash
pip install git+https://github.com/huggingface/transformers@v0.1.10
```

然后，运行以下命令以创建最终的工作权重：

```bash
python -m fastchat.model.apply_delta --base /path/to/llama-13b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13b-delta-v0/
```

现在，您可以准备好使用Vicuna权重了！


最终得到的权重文件应该放在一个文件夹内，具有以下结构：

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

然后，在模型配置文件[here](minigpt4/configs/models/minigpt4.yaml#L16)的第16行设定vicuna权重的路径。

准备预训练的MiniGPT-4检查点

要使用我们预训练的模型，请先下载预训练的检查点，可以通过以下链接下载：

[https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)

然后，在[eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10)中的第11行设置预训练检查点的路径。

### 在本地启动演示

使用以下命令在本地机器上运行[demo.py](demo.py)以尝试我们的演示：

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml
```

### 训练
MiniGPT-4的训练包含两个对齐阶段。

**1. 第一阶段预训练**

在第一阶段预训练中，使用来自Laion和CC数据集的图像-文本对进行训练，以将视觉和语言模型进行对齐。要下载和准备数据集，请参见[第一阶段数据集准备说明](dataset/README_1_STAGE.md)。在第一阶段之后，视觉特征被映射并且可以被语言模型理解。

要启动第一阶段训练，请运行以下命令。在我们的实验中，我们使用了4个A100。您可以在配置文件[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage1_pretrain.yaml)中更改保存路径：

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

第二阶段微调

在第二阶段，我们使用自己创建的小型高质量图像文本对数据集，并将其转换为对话格式以进一步对齐MiniGPT-4。要下载和准备我们的第二阶段数据集，请参见[第二阶段数据集准备说明](dataset/README_2_STAGE.md)。要启动第二阶段对齐，请首先在[train_configs/minigpt4_stage2_finetune.yaml](train_configs/minigpt4_stage2_finetune.yaml)中指定第1阶段中训练的检查点文件路径。您还可以在那里指定输出路径。然后，运行以下命令。在我们的实验中，我们使用了1个A100。

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

第二阶段对齐后，MiniGPT-4能够以连贯且用户友好的方式谈论图像。

## 致谢

+ [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) ：MiniGPT-4的模型架构遵循BLIP-2。如果您以前不知道它，请不要忘记检查这个伟大的开源工作！
+ [Lavis](https://github.com/salesforce/LAVIS) ：这个存储库是基于Lavis构建的！
+ [Vicuna](https://github.com/lm-sys/FastChat) ：只有13B个参数的Vicuna的神奇语言能力真是太棒了。它是开源的！

如果您在研究或应用中使用MiniGPT-4，请引用以下BibTeX：

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## 国内交流群
群主会不定期发布 各类亮眼项目体验版本 供大家体验，星球主要沉淀一些AI知识，帮助节约时间。欢迎各位读者老爷，漂亮姐姐给我的项目点赞！

|              直接加群               |                 如果前面的过期，加我拉你入群                  |                      知识星球                       |
|:-------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|
| <img src="./img/WechatIMG88.jpeg" width="300"/> | <img src="./img/WechatIMG87.jpeg" width="300"/> | <img src="./img/WechatIMG81.jpeg" width="300"/> |

## 许可证.  
此存储库采用[BSD 3-Clause许可证](LICENSE.md)。   
许多代码基于[Lavis](https://github.com/salesforce/LAVIS)，这里是BSD 3-Clause许可证[here](LICENSE_Lavis.md)。   

## 感谢
本项目fork 自 https://github.com/Vision-CAIR/MiniGPT-4   
翻译大部分来自 https://github.com/Vision-CAIR/MiniGPT-4

