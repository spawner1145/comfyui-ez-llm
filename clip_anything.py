import os
import torch
import folder_paths
from transformers import AutoTokenizer, AutoModel
import comfy.sd
import comfy.sd1_clip
import comfy.model_management
import comfy.model_patcher
import comfy.hooks
import logging

llm_dir = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir)

if "LLM" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["LLM"] = ([llm_dir], {".safetensors", ".bin", ".pt"})

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
    """扫描 models/LLM 目录下的所有文件夹"""
    llm_path = os.path.join(folder_paths.models_dir, "LLM")
    if not os.path.exists(llm_path):
        return []
    
    models = []
    for item in os.listdir(llm_path):
        item_path = os.path.join(llm_path, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, "config.json")):
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
            torch_dtype=dtype_torch if dtype_torch else torch.float32,
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
            
            # 使用倒数第二层 (适用于大多数 LLM)
            hidden_states = outputs.hidden_states[-2]
            cond = hidden_states * attention_mask.unsqueeze(-1)
        
        return ([[cond, {"attention_mask": attention_mask}]], )

class LLMTokenizerComfy(comfy.sd1_clip.SDTokenizer):
    def __init__(self, hf_tokenizer, system_prompt, embedding_directory=None, max_length=512):
        self.tokenizer = hf_tokenizer
        self.system_prompt = system_prompt

        self.use_chat_template = hasattr(hf_tokenizer, 'chat_template') and hf_tokenizer.chat_template is not None
        if self.use_chat_template:
            print(f"检测到 chat_template，将使用模板模式")
        else:
            print(f"未检测到 chat_template，将使用文本拼接模式")
        
        self.max_length = max_length if max_length > 0 else 99999999
        self.min_length = 1
        self.end_token = None
        self.min_padding = None
        
        # 检测 special tokens
        empty = self.tokenizer('')["input_ids"]
        
        # 根据 tokenizer 配置 start/end token
        if hasattr(hf_tokenizer, 'bos_token_id') and hf_tokenizer.bos_token_id is not None:
            self.tokens_start = 1
            self.start_token = hf_tokenizer.bos_token_id
            self.tokenizer_adds_end_token = True
            if hasattr(hf_tokenizer, 'eos_token_id') and hf_tokenizer.eos_token_id is not None:
                self.end_token = hf_tokenizer.eos_token_id
        else:
            self.tokens_start = 0
            self.start_token = None
            self.tokenizer_adds_end_token = False
        
        # pad_token
        if hasattr(hf_tokenizer, 'pad_token_id') and hf_tokenizer.pad_token_id is not None:
            self.pad_token = hf_tokenizer.pad_token_id
        elif self.end_token is not None:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0
        
        self.pad_with_end = False  # LLM 通常不用 end token padding
        self.pad_to_max_length = False  # 不强制 padding
        
        # 词汇表
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # embedding 相关
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = 2304  # 默认值，实际不影响
        self.embedding_key = 'llm'
    
    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        """
        选择模板模式(如果有的话)或文本拼接模式
        """
        if self.use_chat_template:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
                # 只获取格式化后的文本，不直接 tokenize
                full_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=False,  # 不添加生成提示（因为只是在编码，不是生成）
                    tokenize=False  # 返回文本而不是 token IDs
                )
            except Exception as e:
                print(f"chat_template 应用失败，降级到文本拼接: {e}")
                full_prompt = f'{self.system_prompt} <Prompt Start> {text}'
        else:
            # 文本拼接模式
            full_prompt = f'{self.system_prompt} <Prompt Start> {text}'
        
        return super().tokenize_with_weights(full_prompt, return_word_ids, disable_weights=False, **kwargs)
    
    def state_dict(self):
        """ComfyUI 需要的方法 (保存 tokenizer 状态)"""
        return {}


class LLMTextEncoderComfy(comfy.sd1_clip.ClipTokenWeightEncoder):
    def __init__(self, hf_model, hf_tokenizer, device="cpu", dtype=None, target_hidden_size=None):
        super().__init__()
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.device = device
        self.dtype = dtype
        self.dtypes = [dtype] if dtype else [torch.float32]
        
        if hasattr(hf_model, 'config'):
            self.num_layers = getattr(hf_model.config, 'num_hidden_layers', 26)
            self.hidden_size = getattr(hf_model.config, 'hidden_size', 2304)
        else:  # 默认值
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
                    torch.nn.init.orthogonal_(self.projection.weight)
            self.projection = self.projection.to(device)
            if dtype:
                self.projection = self.projection.to(dtype)
        
        self.special_tokens = self._get_special_tokens()
        
        # CLIP options (用于 layer 选择等)
        self.layer = "hidden"
        self.layer_idx = -2  # 默认倒数第二层，懒得写到前端，懂的人自己改吧()
        self.return_projected_pooled = False  # LLM 通常不需要 projected pooled

        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)
    
    def _get_special_tokens(self):
        """从 tokenizer 自动获取 special tokens"""
        special_tokens = {}
        
        # 尝试获取各种 special tokens
        if hasattr(self.hf_tokenizer, 'bos_token_id') and self.hf_tokenizer.bos_token_id is not None:
            special_tokens["start"] = self.hf_tokenizer.bos_token_id
        
        if hasattr(self.hf_tokenizer, 'eos_token_id') and self.hf_tokenizer.eos_token_id is not None:
            special_tokens["end"] = self.hf_tokenizer.eos_token_id
        
        if hasattr(self.hf_tokenizer, 'pad_token_id') and self.hf_tokenizer.pad_token_id is not None:
            special_tokens["pad"] = self.hf_tokenizer.pad_token_id
        else:
            # 如果没有 pad_token，使用 eos_token
            special_tokens["pad"] = special_tokens.get("end", 0)
        
        # 如果都没有，使用通用默认值
        if not special_tokens:
            special_tokens = {"start": 2, "end": 1, "pad": 0}
        
        return special_tokens
    
    def reset_clip_options(self):
        """ComfyUI CLIP 需要的方法"""
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]
    
    def set_clip_options(self, options):
        """ComfyUI CLIP 需要的方法，实现 layer 选择"""
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        
        # 验证 layer_idx
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
            self.layer_idx = -1
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx
    
    def gen_empty_tokens(self, special_tokens, length):
        """ClipTokenWeightEncoder 需要的方法，生成空 token 序列"""
        return comfy.sd1_clip.gen_empty_tokens(special_tokens, length)
    
    def state_dict(self):
        """ModelPatcher 需要的方法，返回模型的 state_dict"""
        return self.hf_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载 state_dict（用于 LoRA 等）"""
        return self.hf_model.load_state_dict(state_dict, strict=False)
    
    def to(self, device):
        """ModelPatcher 需要的方法，移动模型到指定设备"""
        self.hf_model = self.hf_model.to(device)
        if self.projection is not None:
            self.projection = self.projection.to(device)
        self.device = device
        return self
    
    def named_modules(self):
        """ModelPatcher 需要的方法，返回所有子模块"""
        return self.hf_model.named_modules()
    
    def parameters(self):
        """返回模型参数"""
        return self.hf_model.parameters()
    
    def named_parameters(self):
        """返回模型参数（带名称）"""
        return self.hf_model.named_parameters()
    
    def encode(self, tokens_list):
        """
        ClipTokenWeightEncoder 需要的方法，只处理 tokens，不处理 weights
        ClipTokenWeightEncoder.encode_token_weights() 会自动调用这个方法并处理权重
        
        输入: tokens_list (list of list of token_id)
        输出: (cond, pooled) 或 (cond, pooled, extra_dict)
        """
        input_ids = torch.tensor(tokens_list, dtype=torch.long).to(self.hf_model.device)
        attention_mask = (input_ids != self.special_tokens.get("pad", 0)).long()
        with torch.no_grad():
            outputs = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 根据 layer_idx 选择输出层
            if self.layer == "last" or self.layer_idx == -1:
                hidden_states = outputs.last_hidden_state
            else:
                # hidden_states 是 tuple，索引从 0 开始
                # layer_idx 可以是负数（从后往前）或正数（从前往后）
                hidden_states = outputs.hidden_states[self.layer_idx]
            
            # attention mask
            cond = hidden_states * attention_mask.unsqueeze(-1)
            
            # 维度投影（神人功能XD）
            if self.projection is not None:
                # [batch, seq_len, hidden_size] -> [batch, seq_len, target_hidden_size]
                original_shape = cond.shape
                cond = self.projection(cond)
                print(f"应用维度投影: {original_shape} -> {cond.shape}")
        
        # 计算 pooled
        if self.return_projected_pooled:
            # 如果需要 projected pooled，使用第一个 token (CLS/BOS)
            pooled = cond[:, 0, :]
        else:
            # 默认：平均池化（忽略 padding）
            mask_expanded = attention_mask.unsqueeze(-1).expand(cond.size()).float()
            sum_embeddings = torch.sum(cond * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        # 返回结果（ClipTokenWeightEncoder 会负责移动到 intermediate_device 和处理 weights）
        return cond, pooled, {"attention_mask": attention_mask}


class LLMCLIPLoader:
    """
    从 Hugging Face 仓库加载任意 LLM，返回真正的 comfy.sd.CLIP 实例
    直接包装 HF 的 model 和 tokenizer，不需要转换 state_dict
    """
    
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
                "device": (devices, {"default": "cpu"}),
                "dtype": (list(DTYPES.keys()), {"default": "default"}),
                "target_hidden_size": ("INT", {
                    "default": 2304, 
                    "min": 512, 
                    "max": 8192, 
                    "step": 128,
                    "tooltip": "目标隐藏层大小。Gemma-2 9B=2304, Qwen-3=1024, 如果与模型不匹配将自动添加投影层"
                }),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "LLM/text encoder"
    TITLE = "Load LLM as CLIP (Universal)"
    DESCRIPTION = "从 HF 仓库加载任意 LLM，输出真正的 ComfyUI CLIP 接口"

    def load_clip(self, model_folder, system_prompt, device, dtype, target_hidden_size=2304):
        dtype_torch = DTYPES[dtype]
        if device == "cpu" and dtype_torch not in [None, torch.float32]:
            raise ValueError(f"CPU 只支持 FP32 或 default! 当前: {dtype}")
        
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_folder)
        if not os.path.exists(model_path):
            raise ValueError(f"模型目录不存在: {model_path}")
        
        print(f"[LLM CLIP] Loading from {model_path}...")
        print(f"[LLM CLIP] Target hidden size: {target_hidden_size}")
        
        # 加载 HF 模型和 tokenizer (与 LLMLoader 相同逻辑)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_tokenizer.padding_side = "right"
        
        hf_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype_torch if dtype_torch else torch.float32,
            device_map=device if device != "cpu" else None,
        )
        
        hf_model.eval()
        hf_model.requires_grad_(False)
        if device != "auto" and device != "cpu":
            hf_model = hf_model.to(device)
        
        if dtype_torch and device != "auto":
            hf_model = hf_model.to(dtype_torch)
        
        # 包装为 ComfyUI 接口
        # 使用较大的 max_length（与 Lumina2/Qwen 等 LLM text encoders 对齐）
        # 默认 512，可以根据模型调整（Lumina2 实际不限制长度，使用 99999999）
        comfy_tokenizer = LLMTokenizerComfy(hf_tokenizer, system_prompt, max_length=512)
        comfy_text_encoder = LLMTextEncoderComfy(
            hf_model, 
            hf_tokenizer, 
            device=device, 
            dtype=dtype_torch,
            target_hidden_size=target_hidden_size
        )
        
        clip = comfy.sd.CLIP(no_init=True)
        clip.cond_stage_model = comfy_text_encoder
        clip.tokenizer = comfy_tokenizer
        
        # 设置 patcher
        load_device = torch.device(device) if device not in ["auto", "cpu"] else comfy.model_management.text_encoder_device()
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
        
        # 从 comfy.sd.CLIP 类获取所有需要的方法
        CLIP_class = comfy.sd.CLIP
        
        # 绑定所有必需的方法
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
        
        # encode_from_tokens_scheduled 如果存在也绑定
        if hasattr(CLIP_class, 'encode_from_tokens_scheduled'):
            clip.encode_from_tokens_scheduled = types.MethodType(CLIP_class.encode_from_tokens_scheduled, clip)
        
        print(f"[LLM CLIP] Loaded as comfy.sd.CLIP!")
        
        return (clip,)
