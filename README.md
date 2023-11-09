# llama2_chat_fine
本项目是一个对llama2进行微调的项目，我是一个llm的初学者，发现网上关于大模型的开源项目和
微调，往往针对于技术层面，但对于初学者，最难的往往是入门的细节方面，本项目基于llama-recipes，
 meta官方的微调教程编写，修改了llama-recipes一些不合理的地方及错误，是一个不错的微调入门项目。

## 下载模型
任何微调项目，第一步一定是下载模型。在这里需要简单的介绍下[llama2](https://github.com/facebookresearch/llama)，
llama2是meta的一个基于transformer的开源大模型项目，且可以商用，所以可以放心使用，它主要分为两种，
一种是llama7B，llama13B，llama70B，另一种是Llama 2-Chat， 是基于Llama 2 针对聊天对话场景微调的版本，使用
SFT (监督微调) 和 RLHF (人类反馈强化学习)进行迭代优化，以便更好的和人类偏好保持一致，提高安全性。这里如果想进行微调的话，
我的建议是Llama2_chat_7b，他的对话能力比Llama2强很多。效果也更好。

**目前下载模型主要分三种方式**

* 去[llama2](https://github.com/facebookresearch/llama)的github项目下clone项目，运行download.sh文件，他会问你选择下载哪个模型，
包括我上面提到的6种，下载的时候需要你提供一段代码凭证，这个凭证需要去[meta官网](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)申请，
申请的速度很快，有了这个申请以后，就可以把模型下载下来，这样虽然能运行， 但是微调起来很不方便，原因为 
  1.微调大家习惯使用的是huggingface模型，需要将下载的模型转换为huggingface形式，相关方法在[llama-recipes](https://github.com/facebookresearch/llama-recipes/)
中有
  2.运行download.sh可能会有很多问题，包括代理，wget等
  3.没有tokenizer.model

* 第二种方法为去huggingface上下载模型，但是如果你直接在huggingface上下载模型，是没有资格的，你需要先按照上面的教程申请凭证，申请以后，用你在meta上
注册的邮箱再注册一个huggingface账号，如果已经有了， 就不用再注册。在huggingface上搜索llama2模型，他会提示你需要凭证， 填写相关内容后， 
需要一段时间，可能几个小时，就可以在huggingface上下载相关的模型了。（推荐这种，但是速度很慢，中国可能需要数个小时甚至10个小时）

* 现在网上已经有很多人上传了模型，可以直接下载别人上传的。

## clone微调代码
模型下载以后，就可以在网上下载微调的代码，网上微调的代码很多，针对中文的也不少，比较典型的项目是[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)等。
这里可以直接拉取我的代码
```
git clone https://github.com/lizhongyi123/llama2_chat_fine.git
```
当然，也可以拉取[llama-recipes](https://github.com/facebookresearch/llama-recipes/)中的代码，官方代码内容更多，且有更多的例子，
我的代码主要是提取出了llama-recipes中我需要的一部分。

## 运行微调代码
当模型和代码都下载下来以后，需要先把代码运行起来，已检查是否有错误，

1. 第一步是安装依赖，
```
pip install -r requirements.txt
```
在运行代码的过程中，有一个windows下才会遇到的问题，bitsandbytes 库会报错，且无法运行，运行以下命令，安装可以在windows运行的库，
```
pip install bitsandbytes-windows
```
如果还是不行，就卸载所有的库
```
pip uninstall bitsandbytes
pip uninstall bitsandbytes-windows
```
然后重新安装
```
pip install bitsandbytes
pip install bitsandbytes-windows
```
注意，本人实践中发现，上面的安装顺序可能对结果产生影响，先安装bitsandbytes-windows发现不起作用，因为本人对bitsandbytes没有进行深入
了解，此处提供的是经验性的建议，如遇问题，可在网上寻找解决方案，亦可咨询本人。

2. 运行默认对话示例  
运行对话示例有两种方法
```angular2html
python chat_completion.py --model_name chat_7b_hf --prompt_file chat.json
```
这是运行我chat.json中的例子， ”what is the recipe of mayonnaise“，
第二种方法为
```angular2html
python chat_completion.py --model_name chat_7b_hf
```
然后在终端输入你的问题，这也是用来检测模型是否能正确回答问题的方法。

3 下载数据集(此处仅做说明，不需要读者编写任何代码)  
这里以项目中默认的samsum数据集为例，项目中下载数据集的代码为
```
 dataset = datasets.load_dataset("samsum", split=split)
```
这段代码不需要你单独运行， 启动项目后，会自动下载。 实践中发现，当pyhton的版本小于3.10.5的时候，上面的命令是没有效果的，可能是python版本太低，同时，
如果你不想将数据集保存到默认路径，调用这个数据集的时候，请使用绝对路径，相对路径极容易报错。再次强调，这段代码是Python项目中的代码， 不需要
读者任何操作，仅作说明。

第二种下载huggingface数据集的方式为huggingface网站手动下载，但是手动下载下来的数据集可能无法直接使用，需要进行一定的转换，这在
网上有很多相关教程

第三种为使用git
```
git lfs install
git clone ...
```
需要注意的是，使用这种方法，在bash中运行的时候，会弹出一个需要你填写账号密码的窗口，这里强调的是，这里的密码不是你的账号密码，
而是你在huggingfaceface 的setting中设置的token，这可以去huggingface上查找相关教程，切记。

第四种目前国内有一些镜像网站保存了一些模型和数据集，尤其是中文，下载速度更快

4 运行默认参数的微调代码
下面的代码是在单节点单单GPU上运行的代码
```
python -m finetuning  --use_peft --peft_method lora --quantization --model_name chat_7b_hf  --output_dir PEFT_model
```
如果在多节点或者多GPU上运行, 下面是一个4GPU运行的例子
```

torchrun --nnodes 1 --nproc_per_node 4  finetuning.py --enable_fsdp --use_peft --peft_method lora 
--model_name chat_7b_hf --pure_bf16 --output_dir PEFT_model
```
如果你在windows上运行上面的代码，这样是不行的，这也是很多人会说不支持在windows上运行llama2的主要原因，下面是llama-recipes官方的代码
```
dist.init_process_group("nccl")
```
nccl是一种只支持linux的分布式方式，我这里改成了windows支持的mpi，其他的选择还有很多，尽管这样也能在windows上跑，但真正训练或者微调大模型，还是
在linux上比较方便，


这里可以简单的介绍下上面代码的含义, --peft_method lora是指在微调方法上选择lora， lora是一种常用的微调方法， --model_name chat_7b_hf
是模型的位置，tokenizer模型也放在这里，在某些模型这样可能不行，这里没问题，--output_dir PEFT_model是微调后模型保存在哪里。在我的项目中，
这些文件都是空的，把模型下载后放在里面就可以使用了。

运行单节点单单GPU上的命令，这是采用代码中默认的samsum数据里来进行微调，samsum是一个较小的总结类数据集，我使用rtx 4090就可以进行微调。



5 合并模型  
如果已经运行成功了上面的例子，你会发现chat_7b_hf是原来的模型，peft_model中是微调得到的模型，存在两个模型，该怎么办呢，
这里可以对lora微调做一个简单的讲解，假设不考虑激活函数这种非线性的作用， 
神经网络可以简化为 
```angular2html
结果 = 特征 * 矩阵A * 矩阵B * 矩阵C...
```
事实上，对于一个十亿甚至百亿级别的代码，每次微调都要调整全部的代码，不但花费巨大，甚至效果也不好，科学家发现每次微调可以直接调整
一部分参数矩阵， 所以对于上面的例子，我只修改矩阵B，得到一个矩阵B_lora, 然后将矩阵B和矩阵B_lora以某种方式加到一起， 得到新
矩阵B， 那么新的神经网络可以表示为
```angular2html
结果 = 特征 * 矩阵A * 新矩阵B * 矩阵C...
```
参考上面的例子，可以直接运行该项目中的merge.py，把其中的路径修改为使用者对应的项目，最好使用绝对路径，将合并好的新模型保存在
new_mode文件夹中，记得把tokenizer依赖的文件也保存进来，这样就可以运行下面的代码。采用lora微调，7B模型按照我的设置，每次实际训练的模型只有400万参数。

6 测试新模型
```angular2html
python chat_completion.py --model_name new_model --prompt_file chat1.json
```
这里为了看微调的结果如何，我问了一个数据集中已经存在的例子，尝试测试别的问题

# 编写中文微调代码
llama2中使用的数据集主要都是英文的，而如果我们想训练一个中文的大模型，应该怎么做呢，答案很简单，用中文数据集来训练他。但是我们需要做什么准备呢？

## 分词器(tokenizer)
这里需要给刚开始学习llm的同学们介绍一下tokenizer， 
通常情况下，Tokenizer有三种粒度：word/char/subword

* word: 按照词进行分词，如: I love you. 则根据空格或标点进行分割 [I love you .]
* character：按照单字符进行分词，就是以char为最小粒度。 如：I love you. 则会分割成 [I l o v e y o u.]
* subword：按照词的subword进行分词。如：Today is sunday. 则会分割成 [I lo ve you.]  
上面的三种方法各有优点，按照word分会出现一个问题，如果你只支持英文，你的tokenizer的词表至少要包括数万单词，如果你要包括各种语言，就需要把
unicode码中的所有词，字全都包含进来，这显然不行

按照character来分， 会出现一个问题， 26个字母就可以用于所有的英语， 这样词表小了，但是字母之间的联系也就没了， love 代表的是l o v e，
而和"爱"没有关系。 所以subword方法使用的是最多的， 它可以较好的平衡词表大小与语义表达能力；常见的子词算法有Byte-Pair Encoding (BPE) / Byte-level BPE（BBPE）、

BPE：即字节对编码。其核心思想是从字母开始，不断找词频最高、且连续的两个token合并，直到达到目标词数。  
BBPE：BBPE核心思想将BPE的从字符级别扩展到字节（Byte）级别。BPE的一个问题是如果遇到了unicode编码，基本字符集可能会很大。
BBPE就是以一个字节为一种“字符”，不管实际字符集用了几个字节来表示一个字符。这样的话，基础字符集的大小就锁定在了256（2^8）。
采用BBPE的好处是可以跨语言共用词表，显著压缩词表的大小。

举一个BBPE的例子，中文学习的“习", 他的unicode码为\u4E60，转化成utf-8为E4B9A0，通常表示为“\xE4\xB9\xA0”。
也就是说分词器遇到了”习“，虽然他的词表中没有，但是可以把他识别为\xE4 \xB9 \xA0， 也就是三个字节，
```angular2html
llama2的tokenizer
token.encode('习') = [231, 188, 163 ]

扩充词表以后的新tokenizer
token.encode('习') = [32069]
```
llama2中的分词器就采用的这种技术， 当遇到不在词表中，且无法识别或处理字符时，会用UTF-8编码将这些字符转换成字节表示。所以这里可以得出一个结论，llama2
是可以处理中文的，也就是说，不修改分词器，也可以处理中文。但是，读了上面的介绍，中国人必然会发现一个问题，一个“习”字要占3个字节，那必然导致编码后的
内容很长，而像中文，日文这种基于单个文字的语言， BBPE可能并不是最好的选择，

## 扩充词表
所以在[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)中，作者重新训练了一个分词器，以处理我上面所说的问题，
重新训练分词器是一种很花费时间的操作，我不建议这么操作。

我选用了一种更简单的操作，直接向llama2的分词器中添加中文， 在expand_token.py中，我将unicode码中的中文部分和一部分标点符号加到词表中，
总共有20000多字，这已经包含了大部分中文，运行expand_token.py文件，会将新的分词器保存到new_token文件夹中，这时你会发现，tokenizer.model
大小并没有变，和原来是一样的，唯一的不同是多了一个文件, added_tokens.json，这个文件中多了很多汉字。这也就可以懂得，我的代码只是通过增加一个词表文件，
实现中文识别的。 可以通过expand_token.py的例子可以来查看词表长度和编码结果，想使用新的词表，只需要把增加的文件复制到模型路径中就可以了。

## 代码修改
上面扩充了词表以后，代码中也需要做相应的修改，才可以进行微调

1. 修改模型的vocab
将模型中的config.json的vocab_size修改为
```angular2html
  "vocab_size": 52339
```
2. 修改finetuning.py中的136行
```angular2html
    # model.resize_token_embeddings(len(tokenizer))
```
将这行代码取消注释，这样模型就可以使用新的词表大小进行embedding

3. 将train_utils.py中的85行代码取消注释
```angular2html
# loss.requires_grad_(True)
```
之所以要修改这行代码，是因为我发现修改词表以后，会提示无法进行梯度计算，而输入的模型参数都是记录梯度信息的，只有loss的梯度被改成了false，我在网上
看了很多教程，发现没有人提到这个，不知道是否是我哪里操作不对。

4.准备数据集
这里我们使用中文数据集来对llama2进行微调， 在我的qa_huatuo.py中我处理了一个[医学问答中文数据集](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-GPTShine)，相关的json文件我放在了personal_dataset文件夹中，
读者可以自行下载，放到该文件夹中，

如果你有其他的数据集，需要修改下面的代码datasets.py中的
```angular2html
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "datasets_processing/qa_huatuo.py"
    train_split: str = "train"
    test_split: str = "validation"
```
把file的路径修改成你实际使用的

5.进行微调
```
python -m finetuning --dataset custom_dataset --use_peft --peft_method lora --quantization --model_name input_path  --output_dir output_path
```
上面的input_path和output_path是用户自己定义的模型路径 

6.合并模型
按照前面讲的方法合并模型即可，此处不再赘述

7 测试新模型
同上

# 小结
这是一个基于llama-recipes项目对llama2进行微调的项目，在学习大语言模型微调的相关项目时候，发现针对宏观技术的较多，针对代码细节
的较少，这对于和我一样想学习llm微调的同学们，难免会产生一些困难，故把我学习过程中主要问题和想法纪录在此，希望有幸能帮到一些看到
该项目的人。本人学习AI不足数月，所写代码及观点，难免有误，望读者阅读时，多怀疑，思考。

另外，这只是llm微调最简单的例子，其他如指令精调，推理加速，上下文扩充，模型优化等高级功能，需读者自行阅读其他项目学习。



