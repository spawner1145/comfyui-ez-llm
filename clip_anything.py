import os
import torch
import folder_paths
from transformers import AutoTokenizer, AutoModel
import comfy.sd
import comfy.sd1_clip
import comfy.model_management
import comfy.model_patcher
import comfy.hooks
import comfy.ops
import logging
import re
from typing import List, Tuple

llm_dir = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir)

if "LLM" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["LLM"] = ([llm_dir], {".safetensors", ".bin", ".pt"})

def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result

def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out

def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text

def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text

def parse_prompt_with_comfy_weights(text: str) -> List[Tuple[str, float]]:
    text = escape_important(text)
    parsed = token_weights(text, 1.0)
    result = [(unescape_important(t), w) for t, w in parsed]
    return result

def chunk_weighted_prompt(text: str, max_length: int, tokenizer) -> List[List[Tuple[str, float]]]:
    weighted_segments = parse_prompt_with_comfy_weights(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for segment_text, weight in weighted_segments:
        if not segment_text.strip():
            continue
            
        tokens = tokenizer(segment_text, add_special_tokens=False)["input_ids"]
        segment_length = len(tokens)
        
        if current_length + segment_length > max_length and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        
        if segment_length > max_length:
            for i in range(0, segment_length, max_length):
                chunk_tokens = tokens[i:i+max_length]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append([(chunk_text, weight)])
        else:
            current_chunk.append((segment_text, weight))
            current_length += segment_length
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [[("", 1.0)]]

DTYPES = {
    "default": None,
    "BF16": torch.bfloat16,
    "FP32": torch.float32,
    "FP16": torch.float16,
}

try: 
    torch.float8_e5m2
    DTYPES["FP8_E4M3"] = torch.float8_e4m3fn
    DTYPES["FP8_E5M2"] = torch.float8_e5m2
except AttributeError:
    print("Torch版本过旧,不支持FP8")

def get_llm_models():
    llm_path = os.path.join(folder_paths.models_dir, "LLM")
    if not os.path.exists(llm_path):
        return []
    
    models = []
    for item in os.listdir(llm_path):
        item_path = os.path.join(llm_path, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
            models.append(item)
    
    return models if models else ["(请将模型放到 models/LLM 目录)"]

class LLMLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["auto", "cpu", "cuda"]
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        
        return {
            "required": {
                "model_folder": (get_llm_models(), {"default": "(请将模型放到 models/LLM 目录)"}),
                "device": (devices, {"default": "cpu"}),
                "dtype": (list(DTYPES.keys()), {"default": "default"}),
            }
        }
    
    RETURN_TYPES = ("LLM",)
    FUNCTION = "load_model"
    CATEGORY = "LLM/text encoder"
    TITLE = "LLM Loader (Universal)"

    def load_model(self, model_folder, device, dtype):
        dtype_torch = DTYPES[dtype]
        if device == "cpu" and dtype_torch not in [None, torch.float32]:
            raise ValueError(f"CPU 只支持 FP32 或 default! 当前: {dtype}")
        
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_folder)
        if not os.path.exists(model_path):
            raise ValueError(f"模型目录不存在: {model_path}")
        
        print(f"Loading LLM from {model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "right"
        
        text_encoder = AutoModel.from_pretrained(
            model_path,
            dtype=dtype_torch if dtype_torch else torch.float32,
            device_map=device if device != "cpu" else None,
        )
        
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        if device != "auto" and device != "cpu":
            text_encoder = text_encoder.to(device)
        
        if dtype_torch and device != "auto":
            text_encoder = text_encoder.to(dtype_torch)
        
        return ({
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "device": device,
            "dtype": dtype_torch,
            "model_path": model_path
        },)

class LLMTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True}),
                "LLM": ("LLM",),
                "system_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."
                }),
                "max_token_length": ("INT", {"default": 256, "min": 64, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "LLM/text encoder"
    TITLE = "LLM Text Encode"

    def encode(self, user_prompt, LLM, system_prompt, max_token_length=256):
        tokenizer = LLM["tokenizer"]
        text_encoder = LLM["text_encoder"]

        full_prompt = f'{system_prompt} <Prompt Start> {user_prompt}'

        with torch.no_grad():
            encodings = tokenizer(
                full_prompt,
                max_length=max_token_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                pad_to_multiple_of=8,
            )
            
            input_ids = encodings.input_ids.to(text_encoder.device)
            attention_mask = encodings.attention_mask.to(text_encoder.device)
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            hidden_states = outputs.hidden_states[-2]
            cond = hidden_states * attention_mask.unsqueeze(-1)
        
        return ([[cond, {"attention_mask": attention_mask}]], )

class LLMTokenizerComfy(comfy.sd1_clip.SDTokenizer):
    def __init__(self, hf_tokenizer, system_prompt, embedding_directory=None, max_length=9999999, enable_weights=False):
        self.hf_tokenizer = hf_tokenizer
        self.system_prompt = system_prompt
        self.enable_weights = enable_weights
        self.max_length = max_length

        self.use_chat_template = hasattr(hf_tokenizer, 'chat_template') and hf_tokenizer.chat_template is not None
        if self.use_chat_template:
            print(f"检测到 chat_template，将使用模板模式")
        else:
            print(f"未检测到 chat_template，将使用文本拼接模式")
        
        print(f"权重功能: {'启用 (ComfyUI原版)' if self.enable_weights else '禁用'}")
        
        # 检测 special tokens
        empty = hf_tokenizer('')["input_ids"]
        
        # 根据 tokenizer 配置 start/end token
        if hasattr(hf_tokenizer, 'bos_token_id') and hf_tokenizer.bos_token_id is not None:
            self.tokens_start = 1
            self.start_token = hf_tokenizer.bos_token_id
            self.tokenizer_adds_end_token = True
            if hasattr(hf_tokenizer, 'eos_token_id') and hf_tokenizer.eos_token_id is not None:
                self.end_token = hf_tokenizer.eos_token_id
            else:
                self.end_token = None
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = None
            self.tokenizer_adds_end_token = False
        
        # pad_token
        if hasattr(hf_tokenizer, 'pad_token_id') and hf_tokenizer.pad_token_id is not None:
            self.pad_token = hf_tokenizer.pad_token_id
        elif self.end_token is not None:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0
        
        self.pad_with_end = False
        self.pad_to_max_length = False
        self.min_length = 1
        self.min_padding = None
        
        # 词汇表
        vocab = hf_tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # embedding 相关（保持兼容性）
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = 2304
        self.embedding_key = 'llm'
    
    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        # 应用 system_prompt
        if self.use_chat_template:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
                full_prompt = self.hf_tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=False,
                    tokenize=False
                )
            except Exception as e:
                print(f"chat_template 应用失败，降级到文本拼接: {e}")
                full_prompt = f'{self.system_prompt} <Prompt Start> {text}'
        else:
            full_prompt = f'{self.system_prompt} <Prompt Start> {text}'
        
        if not self.enable_weights:
            tokens = self.hf_tokenizer(full_prompt, add_special_tokens=True)["input_ids"]
            result = [[(t, 1.0, 0) for t in tokens]]
            return result
        
        chunks = chunk_weighted_prompt(full_prompt, self.max_length - 2, self.hf_tokenizer)
        
        batched_tokens = []
        for chunk_idx, chunk in enumerate(chunks):
            batch = []
            
            if self.start_token is not None:
                batch.append((self.start_token, 1.0, 0))
            
            for segment_text, weight in chunk:
                segment_tokens = self.hf_tokenizer(
                    segment_text, 
                    add_special_tokens=False
                )["input_ids"]
                
                batch.extend([(t, weight, chunk_idx + 1) for t in segment_tokens])
            
            if self.end_token is not None:
                batch.append((self.end_token, 1.0, 0))
            
            if self.pad_to_max_length and len(batch) < self.max_length:
                batch.extend([(self.pad_token, 1.0, 0)] * (self.max_length - len(batch)))
            
            batched_tokens.append(batch)
        
        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in batch] for batch in batched_tokens]
        
        return batched_tokens
    
    def state_dict(self):
        return {}

class LLMTextEncoderComfy(torch.nn.Module, comfy.sd1_clip.ClipTokenWeightEncoder):
    def __init__(self, hf_model, hf_tokenizer, device="cpu", dtype=None, target_hidden_size=None, model_options={}):
        torch.nn.Module.__init__(self)
        comfy.sd1_clip.ClipTokenWeightEncoder.__init__(self)
        
        if hasattr(hf_model, 'config'):
            self.num_layers = getattr(hf_model.config, 'num_hidden_layers', 26)
            self.hidden_size = getattr(hf_model.config, 'hidden_size', 2304)
        else:
            self.num_layers = 26
            self.hidden_size = 2304
        
        self.target_hidden_size = target_hidden_size if target_hidden_size else self.hidden_size
        self.projection = None
        
        if self.hidden_size != self.target_hidden_size:
            print(f"检测到维度不匹配: {self.hidden_size} != {self.target_hidden_size}")
            print(f"创建投影层: Linear({self.hidden_size}, {self.target_hidden_size})")
            self.projection = torch.nn.Linear(self.hidden_size, self.target_hidden_size, bias=False)
            
            with torch.no_grad():
                if self.hidden_size < self.target_hidden_size:
                    self.projection.weight[:self.hidden_size, :] = torch.eye(self.hidden_size)
                else:
                    if dtype == torch.bfloat16:
                        orig_dtype = self.projection.weight.dtype
                        self.projection.weight.data = self.projection.weight.data.to(torch.float32)
                        torch.nn.init.orthogonal_(self.projection.weight)
                        self.projection.weight.data = self.projection.weight.data.to(orig_dtype)
                    else:
                        torch.nn.init.orthogonal_(self.projection.weight)
            
            self.projection = self.projection.to(device)
            if dtype:
                self.projection = self.projection.to(dtype)
        
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.transformer = hf_model
        self.device = device
        self.dtype = dtype
        self.dtypes = [dtype] if dtype else [torch.float32]
        
        self.max_length = 512
        self.layer = "hidden"
        self.layer_idx = -2
        self.return_projected_pooled = False
        self.special_tokens = self._get_special_tokens()
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)
        
        self.enable_attention_masks = True
        self.return_attention_masks = True
        self.layer_norm_hidden_state = False
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.operations = model_options.get("custom_operations", comfy.ops.manual_cast)
    
    def _get_special_tokens(self):
        special_tokens = {}
        
        if hasattr(self.hf_tokenizer, 'bos_token_id') and self.hf_tokenizer.bos_token_id is not None:
            special_tokens["start"] = self.hf_tokenizer.bos_token_id
        
        if hasattr(self.hf_tokenizer, 'eos_token_id') and self.hf_tokenizer.eos_token_id is not None:
            special_tokens["end"] = self.hf_tokenizer.eos_token_id
        
        if hasattr(self.hf_tokenizer, 'pad_token_id') and self.hf_tokenizer.pad_token_id is not None:
            special_tokens["pad"] = self.hf_tokenizer.pad_token_id
        else:
            special_tokens["pad"] = special_tokens.get("end", 0)
        
        if not special_tokens:
            special_tokens = {"start": 2, "end": 1, "pad": 0}
        
        return special_tokens
    
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]
    
    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
            self.layer_idx = None
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx
    
    def gen_empty_tokens(self, special_tokens, length):
        return comfy.sd1_clip.gen_empty_tokens(special_tokens, length)
    
    def state_dict(self):
        return self.hf_model.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.hf_model.load_state_dict(state_dict, strict=False)
    
    def to(self, device):
        self.hf_model = self.hf_model.to(device)
        if self.projection is not None:
            self.projection = self.projection.to(device)
        self.device = device
        return self
    
    def named_modules(self):
        return self.hf_model.named_modules()
    
    def parameters(self):
        return self.hf_model.parameters()
    
    def named_parameters(self):
        return self.hf_model.named_parameters()
    
    def encode(self, tokens_list):
        input_ids = torch.tensor(tokens_list, dtype=torch.long).to(self.hf_model.device)
        attention_mask = (input_ids != self.special_tokens.get("pad", 0)).long().to(input_ids.device)
        
        with torch.no_grad():
            outputs = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            hidden_states = outputs.hidden_states[-2]

            cond = hidden_states * attention_mask.unsqueeze(-1)
            
            if self.projection is not None:
                original_shape = cond.shape
                cond = self.projection(cond)
                print(f"应用维度投影: {original_shape} -> {cond.shape}")
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(cond.size()).float().to(cond.device)
        sum_embeddings = torch.sum(cond * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        return cond, pooled, {"attention_mask": attention_mask}


class LLMCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["auto", "cpu", "cuda"]
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        
        return {
            "required": {
                "model_folder": (get_llm_models(), ),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."
                }),
                "device": (devices, {"default": "cuda"}),
                "dtype": (list(DTYPES.keys()), {"default": "FP16"}),
                "target_hidden_size": ("INT", {
                    "default": 2304, 
                    "min": 512, 
                    "max": 8192, 
                    "step": 128,
                    "tooltip": "目标隐藏层大小。Gemma-2 2B=2304, Qwen-3=1024"
                }),
                "enable_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用权重功能：支持 (word:1.5) 语法和超长提示词自动分批"
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "强制模型在不使用时卸载到CPU，减少显存占用"
                }),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "LLM/text encoder"
    TITLE = "Load LLM as CLIP (Universal)"

    def load_clip(self, model_folder, system_prompt, device, dtype, target_hidden_size=2304, enable_weights=False, force_offload=True):
        dtype_torch = DTYPES[dtype]
        if device == "cpu" and dtype_torch not in [None, torch.float32]:
            raise ValueError(f"CPU 只支持 FP32 或 default! 当前: {dtype}")
        
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_folder)
        if not os.path.exists(model_path):
            raise ValueError(f"模型目录不存在: {model_path}")
        
        print(f"[LLM CLIP] Loading from {model_path}...")
        print(f"[LLM CLIP] Target hidden size: {target_hidden_size}")
        print(f"[LLM CLIP] Enable weights: {enable_weights}")
        print(f"[LLM CLIP] Force offload: {force_offload}")
        
        # 加载 HF 模型和 tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_tokenizer.padding_side = "right"
        
        # 明确指定设备加载模型
        device_map = None
        if device != "auto" and device != "cpu":
            device_map = {device: 0}
            torch_device = torch.device(device)
        elif device == "cpu":
            torch_device = torch.device("cpu")
        else:  # auto
            torch_device = comfy.model_management.text_encoder_device()
            device = torch_device.type
            if torch_device.index is not None:
                device += f":{torch_device.index}"
        
        # 先在CPU加载模型，再转移到目标设备，确保参数完整性
        hf_model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype_torch if dtype_torch else torch.float32,
            device_map="cpu",  # 先在CPU完整加载
        )
        
        hf_model.eval()
        hf_model.requires_grad_(False)
        
        hf_model = hf_model.to(torch_device)
        if dtype_torch:
            hf_model = hf_model.to(dtype_torch)
        
        if force_offload and device != "cpu":
            class SafeOffloadModelWrapper(torch.nn.Module):
                def __init__(self, model, load_device, offload_device, dtype):
                    super().__init__()
                    self.model = model.to(offload_device)
                    self.load_device = load_device
                    self.offload_device = offload_device
                    self.dtype = dtype
                    self._is_loaded = False
                    self.original_device = load_device

                def _ensure_model_on_device(self):
                    """确保模型在正确的设备上，并验证所有参数都在该设备"""
                    if not self._is_loaded:
                        self.model = self.model.to(self.load_device)
                        if self.dtype:
                            self.model = self.model.to(self.dtype)
                        
                        wrong_device_params = []
                        for name, param in self.model.named_parameters():
                            if param.device != self.load_device:
                                wrong_device_params.append(name)
                        
                        if wrong_device_params:
                            print(f"[LLM CLIP] 发现 {len(wrong_device_params)} 个参数在错误的设备上，正在修复...")
                            for name, param in self.model.named_parameters():
                                if param.device != self.load_device:
                                    param.data = param.data.to(self.load_device)
                                    if self.dtype:
                                        param.data = param.data.to(self.dtype)
                        
                        self._is_loaded = True

                def forward(self, *args, **kwargs):
                    args = [
                        arg.to(self.load_device) if isinstance(arg, torch.Tensor) else arg
                        for arg in args
                    ]
                    
                    kwargs = {
                        k: v.to(self.load_device) if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items()
                    }
                    
                    self._ensure_model_on_device()
                    
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                        first_layer = self.model.model.layers[0]
                        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                            if first_layer.self_attn.q_proj.weight.device != self.load_device:
                                print(f"[LLM CLIP] q_proj 层在错误设备上，正在纠正...")
                                first_layer.self_attn.q_proj = first_layer.self_attn.q_proj.to(self.load_device)
                    
                    result = self.model(*args, **kwargs)

                    # 卸载模型到CPU，但保持输出在GPU上
                    self.model = self.model.to(self.offload_device)
                    self._is_loaded = False
                    
                    return result
                
                def __getattr__(self, name):
                    if name in ['model', 'load_device', 'offload_device', 'dtype', '_is_loaded', 'original_device']:
                        return super().__getattr__(name)
                    return getattr(self.model, name)
            
            hf_model = SafeOffloadModelWrapper(
                hf_model, 
                torch_device, 
                torch.device("cpu"),
                dtype_torch
            )
        
        class FixedLLMTextEncoderComfy(LLMTextEncoderComfy):
            def encode(self, tokens_list):
                target_device = torch_device
                if hasattr(self.hf_model, 'device'):
                    target_device = self.hf_model.device
                elif hasattr(self.hf_model, 'load_device'):
                    target_device = self.hf_model.load_device
                
                input_ids = torch.tensor(tokens_list, dtype=torch.long).to(target_device)
                attention_mask = (input_ids != self.special_tokens.get("pad", 0)).long().to(input_ids.device)
                
                with torch.no_grad():
                    outputs = self.hf_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    
                    hidden_states = outputs.hidden_states[-2].to(input_ids.device)
                    cond = hidden_states * attention_mask.unsqueeze(-1)

                    # 维度投影(神人功能XD)
                    if self.projection is not None:
                        self.projection = self.projection.to(cond.device)
                        original_shape = cond.shape
                        cond = self.projection(cond)
                        print(f"应用维度投影: {original_shape} -> {cond.shape}")
                    else:
                        if cond.shape[-1] != self.target_hidden_size:
                            print(f"[LLM CLIP] 警告: 维度不匹配但未创建投影层，自动创建...")
                            self.projection = torch.nn.Linear(cond.shape[-1], self.target_hidden_size, bias=False).to(cond.device, dtype=cond.dtype)
                            
                            with torch.no_grad():
                                if cond.dtype == torch.bfloat16:
                                    orig_dtype = self.projection.weight.dtype
                                    self.projection.weight.data = self.projection.weight.data.to(torch.float32)
                                    torch.nn.init.orthogonal_(self.projection.weight)
                                    self.projection.weight.data = self.projection.weight.data.to(orig_dtype)
                                else:
                                    torch.nn.init.orthogonal_(self.projection.weight)
                            
                            original_shape = cond.shape
                            cond = self.projection(cond)
                            print(f"[LLM CLIP] 自动创建并应用投影层: {original_shape} -> {cond.shape}")
                
                # 确保所有张量在同一设备上
                mask_expanded = attention_mask.unsqueeze(-1).expand(cond.size()).float().to(cond.device)
                sum_embeddings = torch.sum(cond * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
                
                return cond, pooled, {"attention_mask": attention_mask.to(cond.device)}
        
        comfy_tokenizer = LLMTokenizerComfy(
            hf_tokenizer, 
            system_prompt, 
            max_length=9999999,
            enable_weights=enable_weights
        )
        
        comfy_text_encoder = FixedLLMTextEncoderComfy(
            hf_model, 
            hf_tokenizer, 
            device=torch_device, 
            dtype=dtype_torch,
            target_hidden_size=target_hidden_size
        )
        
        clip = comfy.sd.CLIP(no_init=True)
        clip.cond_stage_model = comfy_text_encoder
        clip.tokenizer = comfy_tokenizer
        
        # 设置 patcher
        load_device = torch_device
        offload_device = comfy.model_management.text_encoder_offload_device()
        
        clip.patcher = comfy.model_patcher.ModelPatcher(
            comfy_text_encoder,
            load_device=load_device,
            offload_device=offload_device
        )
        clip.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        clip.patcher.is_clip = True
        clip.layer_idx = None
        clip.use_clip_schedule = False
        clip.tokenizer_options = {}
        clip.apply_hooks_to_conds = None
        
        # 直接从 comfy.sd.CLIP 类复制方法（防止写石山这一块）
        # 这样确保 100% 兼容，因为使用的是完全相同的实现
        import types
        CLIP_class = comfy.sd.CLIP
        
        clip.load_model = types.MethodType(CLIP_class.load_model, clip)
        clip.clone = types.MethodType(CLIP_class.clone, clip)
        clip.add_patches = types.MethodType(CLIP_class.add_patches, clip)
        clip.set_tokenizer_option = types.MethodType(CLIP_class.set_tokenizer_option, clip)
        clip.clip_layer = types.MethodType(CLIP_class.clip_layer, clip)
        clip.tokenize = types.MethodType(CLIP_class.tokenize, clip)
        clip.add_hooks_to_dict = types.MethodType(CLIP_class.add_hooks_to_dict, clip)
        clip.encode_from_tokens = types.MethodType(CLIP_class.encode_from_tokens, clip)
        clip.encode = types.MethodType(CLIP_class.encode, clip)
        clip.load_sd = types.MethodType(CLIP_class.load_sd, clip)
        clip.get_sd = types.MethodType(CLIP_class.get_sd, clip)
        clip.get_key_patches = types.MethodType(CLIP_class.get_key_patches, clip)
        
        if hasattr(CLIP_class, 'encode_from_tokens_scheduled'):
            clip.encode_from_tokens_scheduled = types.MethodType(CLIP_class.encode_from_tokens_scheduled, clip)
        
        print(f"[LLM CLIP] Loaded as comfy.sd.CLIP!")
        
        return (clip,)
