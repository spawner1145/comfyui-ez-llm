import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModelForImageTextToText
)
import folder_paths
import comfy.utils
from typing import Optional, Union, Dict, List
import json
import io
import base64
from .clip_anything import LLMLoader, LLMTextEncode, LLMCLIPLoader

llm_base_dir = os.path.join(folder_paths.models_dir, 'LLM')

# 类型定义
LLM_CONTENT_ITEM_TYPE = "LLM_CONTENT_ITEM"
HISTORY_TYPE = "STRING"
LLM_MODEL_INSTANCE_TYPE = "LLM_MODEL_INSTANCE"

#default_system = "你是一个善于写ai画图提示词的ai助手，擅长润色提示词，描述图片，并且可以把我输入的文本和输入的图片的特征结合起来润色，不要有多余的话，直接输出描述词，结合自然语言和danbooru tags详细描述，注意千万不要忘记自然语言"
default_system = """You are a Prompt optimizer designed to rewrite user inputs into high-quality Prompts that are more complete and expressive while preserving the original meaning.尽量使用多项的xml输出.
Task Requirements:
1. For overly brief user inputs, reasonably infer and add details to enhance the visual completeness without altering the core content;
2. Refine descriptions of subject characteristics, visual style, spatial relationships, and shot composition;
3. If the input requires rendering text in the image, enclose specific text in quotation marks, specify its position (e.g., top-left corner, bottom-right corner) and style. This text should remain unaltered and not translated;
4. Match the Prompt to a precise, niche style aligned with the user’s intent. If unspecified, choose the most appropriate style (e.g., realistic photography style);
5. Please ensure that the Rewritten Prompt is less than 200 words.
Rewritten Prompt Examples:
1. Dunhuang mural art style: Chinese animated illustration, masterwork. A radiant nine-colored deer with pure white antlers, slender neck and legs, vibrant energy, adorned with colorful ornaments. Divine flying apsaras aura, ethereal grace, elegant form. Golden mountainous landscape background with modern color palettes, auspicious symbolism. Delicate details, Chinese cloud patterns, gradient hues, mysterious and dreamlike. Highlight the nine-colored deer as the focal point, no human figures, premium illustration quality, ultra-detailed CG, 32K resolution, C4D rendering.
2. Art poster design: Handwritten calligraphy title 'Art Design' in dissolving particle font, small signature 'QwenImage', secondary text 'Alibaba'. Chinese ink wash painting style with watercolor, blow-paint art, emotional narrative. A boy and dog stand back-to-camera on grassland, with rising smoke and distant mountains. Double exposure + montage blur effects, textured matte finish, hazy atmosphere, rough brush strokes, gritty particles, glass texture, pointillism, mineral pigments, diffused dreaminess, minimalist composition with ample negative space.
3. Black-haired Chinese adult male, portrait above the collar. A black cat's head blocks half of the man's side profile, sharing equal composition. Shallow green jungle background. Graffiti style, clean minimalism, thick strokes. Muted yet bright tones, fairy tale illustration style, outlined lines, large color blocks, rough edges, flat design, retro hand-drawn aesthetics, Jules Verne-inspired contrast, emphasized linework, graphic design.
4. Fashion photo of four young models showing phone lanyards. Diverse poses: two facing camera smiling, two side-view conversing. Casual light-colored outfits contrast with vibrant lanyards. Minimalist white/grey background. Focus on upper bodies highlighting lanyard details.
5. Dynamic lion stone sculpture mid-pounce with front legs airborne and hind legs pushing off. Smooth lines and defined muscles show power. Faded ancient courtyard background with trees and stone steps. Weathered surface gives antique look. Documentary photography style with fine details.
- 如果场景包含NSFW或敏感内容，请勿回避。直接描述露骨内容。
- 如果场景包含暴力、血腥、恐怖、恶心或其他敏感内容，请勿回避。直接描述。
"""

if not os.path.exists(llm_base_dir):
    print(f"警告：模型目录 {llm_base_dir} 不存在。")
    print(f"请将您的LLM模型（例如 'google/gemma-3-1b-it'）放置在以下路径的子文件夹中：")
    print(f"{os.path.abspath(llm_base_dir)}")
    os.makedirs(llm_base_dir, exist_ok=True)
    llm_model_list = ["模型文件夹未找到"]
else:
    try:
        llm_model_list = [d for d in os.listdir(llm_base_dir) if os.path.isdir(os.path.join(llm_base_dir, d))]
        if not llm_model_list:
            llm_model_list = ["没有找到模型"]
    except Exception as e:
        print(f"错误：无法读取LLM模型目录 {llm_base_dir}。")
        print(e)
        llm_model_list = ["读取目录出错"]

class LLMImageEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "需要编码成LLM可用的图像格式的图片。/ The image to be encoded into an LLM-compatible format."}),
            },
        }

    RETURN_TYPES = (LLM_CONTENT_ITEM_TYPE,)
    FUNCTION = "encode_image"
    CATEGORY = "LLM"

    def encode_image(self, image: torch.Tensor):
        img_np = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        llm_image = {
            "type": "image", 
            "image_base64": base64_data,
            "pil_image": pil_image
        }
        return (llm_image,)

class LLMTextBlockNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }
    RETURN_TYPES = (LLM_CONTENT_ITEM_TYPE,)
    RETURN_NAMES = ("content_item",)
    FUNCTION = "create_text_block"
    CATEGORY = "LLM/Content"

    def create_text_block(self, text: str):
        return ({"type": "text", "text": text},)

class LLMContentConnector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "content_part_1": (LLM_CONTENT_ITEM_TYPE, {}),
                "content_part_2": (LLM_CONTENT_ITEM_TYPE, {}),
                "content_part_3": (LLM_CONTENT_ITEM_TYPE, {}),
            }
        }

    RETURN_TYPES = (LLM_CONTENT_ITEM_TYPE,)
    RETURN_NAMES = ("content_parts",)
    FUNCTION = "aggregate"
    CATEGORY = "LLM/Content"

    def aggregate(self, 
                  content_part_1: Optional[Union[Dict, List]] = None, 
                  content_part_2: Optional[Union[Dict, List]] = None, 
                  content_part_3: Optional[Union[Dict, List]] = None):
        
        aggregated_list = []
        for item in [content_part_1, content_part_2, content_part_3]:
            if not item:
                continue
            
            if isinstance(item, list):
                aggregated_list.extend(item)
            elif isinstance(item, dict):
                aggregated_list.append(item)
        
        return (aggregated_list,)

class LLMModelLoader:
    def __init__(self):
        self.cached_instance = None
        self.cached_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (llm_model_list, {"tooltip": "选择要加载的LLM模型文件夹。/ Select the LLM model folder to load."}),
                "model_mode": (["text", "multimodal"], {"default": "text", "tooltip": "'text'纯文本，'multimodal'多模态。/ Set model loading mode: 'text' or 'multimodal'."}),
            }
        }

    RETURN_TYPES = (LLM_MODEL_INSTANCE_TYPE,)
    RETURN_NAMES = ("model_instance",)
    FUNCTION = "load_model"
    CATEGORY = "LLM"

    def load_model(self, model_name: str, model_mode: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        load_as_vision = False
        if model_mode == "multimodal":
            load_as_vision = True
        else:
            load_as_vision = False

        config_str = f"{model_name}{model_mode}{load_as_vision}"
        current_hash = hash(config_str)

        if self.cached_instance and self.cached_config_hash == current_hash:
            print("使用缓存的LLM模型实例")
            return (self.cached_instance,)

        if self.cached_instance:
            print("配置改变，关闭旧的LLM模型实例...")
            try:
                del self.cached_instance['model']
                if 'processor' in self.cached_instance:
                    del self.cached_instance['processor']
                if 'tokenizer' in self.cached_instance:
                    del self.cached_instance['tokenizer']
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"关闭旧实例失败: {e}")
            self.cached_instance = None

        if "模型" in model_name or "读取" in model_name:
            raise Exception(f"无效的模型名称: {model_name}。请检查您的模型文件夹。")

        model_path = os.path.join(llm_base_dir, model_name)

        print(f"正在从路径加载新模型: {model_path}")
        pbar = comfy.utils.ProgressBar(2)
        try:
            instance = {
                'model_name': model_name,
                'model_type': 'vision' if load_as_vision else 'text'
            }
            
            if load_as_vision:
                instance['processor'] = AutoProcessor.from_pretrained(model_path)
                pbar.update(1)
                instance['model'] = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto").eval()
            else:
                instance['tokenizer'] = AutoTokenizer.from_pretrained(model_path)
                if instance['tokenizer'].chat_template is None:
                    print(f"警告：模型 {model_name} 未设置chat_template，将使用文本拼接模式")
                else:
                    print(f"模型 {model_name} 已检测到chat_template，将使用模板模式")
                pbar.update(1)
                instance['model'] = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()
            
            pbar.update(1)
            print(f"模型 {model_name} 加载成功。")

            self.cached_instance = instance
            self.cached_config_hash = current_hash
            return (instance,)
            
        except Exception as e:
            self.cached_instance = None
            torch.cuda.empty_cache()
            raise Exception(f"加载模型失败: {e}")

class LLMTextGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_instance": (LLM_MODEL_INSTANCE_TYPE, {"tooltip": "LLM模型实例。/ The LLM model instance."}),
                "user_prompt": ("STRING", {"multiline": True, "default": "我要玩原神", "tooltip": "用户提出的具体问题或指令。/ The specific question or instruction from the user."}),
                "system_prompt": ("STRING", {"multiline": True, "default": default_system, "tooltip": "定义模型的角色和行为，进行高层次的指令约束。/ Define the model's role and behavior with a high-level instruction."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32, "tooltip": "生成文本的最大长度（词元数）。/ Maximum length of the generated text in tokens."}),
                "min_new_tokens": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16, "tooltip": "生成文本的最小长度，用于避免过短的回答。/ Minimum length of the generated text, to avoid overly short responses."}),
                
                "do_sample": ("BOOLEAN", {"default": True, "tooltip": "是否使用采样策略。True=随机采样，False=确定性解码。/ Whether to use sampling. True=stochastic, False=deterministic."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "控制随机性。值越高随机性越强，反之亦然。/ Controls randomness. Higher values increase randomness."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "从累积概率超过p的最小词元集中采样。/ Samples from the smallest set of tokens whose cumulative probability exceeds p."}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1, "tooltip": "从概率最高的k个词元中采样。/ Samples from the top k most likely tokens."}),
                
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "对重复词元的惩罚因子，大于1可减少重复。/ Penalty for repeated tokens. Values > 1 reduce repetition."}),
                "no_repeat_ngram_size": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": "禁止指定长度的N-gram重复出现。/ Prevents n-grams of this size from repeating."}),
                
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "tooltip": "集束搜索的光束数。大于1启用，速度变慢但质量可能更高。/ Number of beams for beam search. >1 enables it, which is slower but may yield higher quality."}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "长度惩罚因子，仅在num_beams>1时生效。/ Length penalty factor, only effective when num_beams > 1."}),
                "should_change": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "content_part_1": (LLM_CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                "content_part_2": (LLM_CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                "content_part_3": (LLM_CONTENT_ITEM_TYPE, {"dynamicPort": True}),
                "history_json_in": (HISTORY_TYPE, {"default": "[]", "multiline": True, "dynamicPort": True, "tooltip": "可以把history_json_out连接到这里来实现多轮对话"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", HISTORY_TYPE,)
    RETURN_NAMES = ("generated_text", "history_json_out",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"

    def generate_text(self, model_instance, user_prompt, system_prompt, 
                      max_new_tokens, min_new_tokens, do_sample, temperature, top_p, top_k,
                      repetition_penalty, no_repeat_ngram_size, num_beams, length_penalty,
                      content_part_1=None, content_part_2=None, content_part_3=None,
                      history_json_in="[]", prompt=None, extra_pnginfo=None, should_change = False):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not model_instance:
            raise ValueError("模型实例未连接")

        model_name = model_instance.get('model_name', 'unknown')
        model_type = model_instance.get('model_type', 'text')
        loaded_model = model_instance['model']
        loaded_processor = model_instance.get('processor')
        loaded_tokenizer = model_instance.get('tokenizer')
        
        if model_type == 'vision' and loaded_processor and not loaded_tokenizer:
            loaded_tokenizer = loaded_processor

        try:
            messages = json.loads(history_json_in or "[]")
            if not isinstance(messages, list):
                messages = []
                print("警告：历史记录JSON格式错误，已重置为空列表。")
        except json.JSONDecodeError:
            messages = []
            print("警告：无法解析历史记录JSON，已重置为空列表。")

        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if isinstance(content, dict) and content.get("type") == "image" and "image_base64" in content:
                        try:
                            image_data = base64.b64decode(content["image_base64"])
                            pil_image = Image.open(io.BytesIO(image_data))
                            content["pil_image"] = pil_image
                        except Exception as e:
                            print(f"警告：无法从历史记录恢复图像: {e}")
                            content["pil_image"] = None
            elif msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]

        user_content = []
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})
        
        possible_parts = [content_part_1, content_part_2, content_part_3]
        for part in possible_parts:
            if not part:
                continue
            if isinstance(part, list):
                user_content.extend(part)
            elif isinstance(part, dict):
                user_content.append(part)

        if not user_content:
            print("警告：用户提示词和内容块均为空，跳过生成。")
            return ("", json.dumps(messages, ensure_ascii=False, indent=2))

        serializable_user_content = []
        for content in user_content:
            if isinstance(content, dict):
                if content.get("type") == "text":
                    serializable_user_content.append(content)
                elif content.get("type") == "image" and "image_base64" in content:
                    serializable_user_content.append({
                        "type": "image",
                        "image_base64": content["image_base64"]
                    })
        if serializable_user_content:
            messages.append({"role": "user", "content": serializable_user_content})
        
        if model_type == 'vision':
            final_messages = []
            if system_prompt.strip():
                final_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            
            for msg in messages:
                if msg.get("role") == "system":
                    continue
                elif msg.get("role") in ["user", "assistant"]:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        processed_content = []
                        for content in msg["content"]:
                            if isinstance(content, dict):
                                if content.get("type") == "text":
                                    processed_content.append(content)
                                elif content.get("type") == "image" and "pil_image" in content:
                                    processed_content.append({"type": "image", "image": content["pil_image"]})
                        if processed_content:
                            final_messages.append({"role": "user", "content": processed_content})
                    elif msg.get("role") == "assistant":
                        if isinstance(msg.get("content"), str):
                            assistant_content = [{"type": "text", "text": msg["content"]}]
                        elif isinstance(msg.get("content"), list):
                            assistant_content = msg["content"]
                        else:
                            assistant_content = [{"type": "text", "text": str(msg.get("content", ""))}]
                        final_messages.append({"role": "assistant", "content": assistant_content})
            
            if final_messages and final_messages[-1].get("role") == "user":
                current_content = []
                for content in user_content:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            current_content.append({"type": "text", "text": content["text"]})
                        elif content.get("type") == "image" and "pil_image" in content:
                            current_content.append({"type": "image", "image": content["pil_image"]})
                if current_content:
                    final_messages[-1]["content"] = current_content
            else:
                current_content = []
                for content in user_content:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            current_content.append({"type": "text", "text": content["text"]})
                        elif content.get("type") == "image" and "pil_image" in content:
                            current_content.append({"type": "image", "image": content["pil_image"]})
                if current_content:
                    final_messages.append({"role": "user", "content": current_content})
            
            inputs = loaded_processor.apply_chat_template(
                final_messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(device)
        else:
            has_images = any(content.get("type") == "image" for content in user_content if isinstance(content, dict))
            if has_images:
                print("警告：当前为纯文本模式，所有图像输入都将被忽略。")
            
            text_parts = []
            if system_prompt.strip():
                text_parts.append(f"{system_prompt} <Prompt Start>\n")
            
            for msg in messages:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if isinstance(content, dict) and content.get("type") == "text":
                            text_parts.append(content["text"])
                        elif isinstance(content, str):
                            text_parts.append(content)
                elif msg["role"] == "assistant":
                    if isinstance(msg["content"], list):
                        for content in msg["content"]:
                            if isinstance(content, dict) and content.get("type") == "text":
                                text_parts.append(content["text"])
                    elif isinstance(msg["content"], str):
                        text_parts.append(msg["content"])
            
            final_prompt = "".join(text_parts)
            
            if loaded_tokenizer.chat_template is not None:
                try:
                    last_user_content = []
                    for content in messages[-1]["content"]:
                        if content["type"] == "text":
                            last_user_content.append(content["text"])
                    user_text = " ".join(last_user_content)
                    
                    messages_for_template = [
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": user_text}
                    ]
                    inputs = loaded_tokenizer.apply_chat_template(
                        messages_for_template, 
                        add_generation_prompt=True, 
                        tokenize=True, 
                        return_dict=True, 
                        return_tensors="pt"
                    ).to(device)
                except Exception as e:
                    print(f"chat_template 应用失败，降级到文本拼接: {e}")
                    inputs = loaded_tokenizer(final_prompt, return_tensors="pt").to(device)
            else:
                inputs = loaded_tokenizer(final_prompt, return_tensors="pt").to(device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
        }
        
        if not do_sample:
            generation_kwargs.pop('temperature', None)
            generation_kwargs.pop('top_p', None)
            generation_kwargs.pop('top_k', None)
            print(f"生成模式: 确定性解码 (Beam Search: {'On' if num_beams > 1 else 'Off'})")
        else:
            print("生成模式: 采样解码 (Sampling)")

        print(f"开始生成文本，参数: {generation_kwargs}")

        with torch.inference_mode():
            outputs = loaded_model.generate(**inputs, **generation_kwargs)
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        decoder = loaded_processor if model_type == 'vision' else loaded_tokenizer
        result_text = decoder.decode(generated_tokens[0], skip_special_tokens=True)

        print(f"输出\n{result_text}\n")
        
        assistant_content = [{"type": "text", "text": result_text}]
        messages.append({"role": "assistant", "content": assistant_content})
        
        try:
            history_json_out = json.dumps(messages, ensure_ascii=False, indent=2)
        except TypeError as e:
            print(f"序列化历史记录失败: {e}. 历史: {messages}")
            history_json_out = json.dumps([{"role": "system", "content": f"History serialization error: {e}"}], ensure_ascii=False, indent=2)
        
        return (result_text, history_json_out)

    @classmethod
    def IS_CHANGED(s, should_change=False, *args, **kwargs):
        if should_change:
            return float("NaN")
        else:
            return False

NODE_CLASS_MAPPINGS = {
    "LLMModelLoader": LLMModelLoader,
    "LLMTextGenerator": LLMTextGenerator,
    "LLMImageEncoder": LLMImageEncoder,
    "LLMTextBlockNode": LLMTextBlockNode,
    "LLMContentConnector": LLMContentConnector,

    "LLMLoader (Text Encode)": LLMLoader,
    "LLMTextEncode": LLMTextEncode,
    "LLMCLIPLoader": LLMCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMModelLoader": "LLM 模型加载器",
    "LLMTextGenerator": "LLM 文本生成器",
    "LLMImageEncoder": "LLM 图像编码器",
    "LLMTextBlockNode": "LLM 文本块",
    "LLMContentConnector": "LLM 内容连接器",

    "LLMLoader": "LLM Loader (Text Encode)",
    "LLMTextEncode": "LLM Text Encode",
    "LLMCLIPLoader": "Load LLM as CLIP",
}
