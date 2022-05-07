from transformers import AutoTokenizer,AutoConfig,AutoModel

tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en", cache_dir='./ernie_cache')
