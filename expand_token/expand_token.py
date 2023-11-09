import os
import sys
from transformers import LlamaTokenizer
script_directory = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在文件夹的绝对路径
script_directory = os.path.dirname(script_directory)

model = os.path.join(script_directory, 'chat_7b_hf')
tokenizer = LlamaTokenizer.from_pretrained(model)
#
#
#
#生成包含所有汉字的列表
all_chinese_characters = [chr(i) for i in range(0x4e00, 0x9fff)]

chinese_punctuation =  """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】
〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
chinese_punctuation_list = list(chinese_punctuation)

chinese = all_chinese_characters + chinese_punctuation_list
#
tokenizer.add_tokens(chinese)

path_of_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_token')
tokenizer.save_pretrained(path_of_save)

# new_token = LlamaTokenizer.from_pretrained(path_of_save)
# print(len(new_token))
#
#
# v = new_token.encode('我爱中国')
#
# print(v)


