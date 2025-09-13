import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    Gemma3ForConditionalGeneration # AutoModelForConditionalGeneration
)
import folder_paths
import comfy.utils

llm_base_dir = os.path.join(folder_paths.models_dir, 'LLM')

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

    RETURN_TYPES = ("LLM_IMAGE",)
    FUNCTION = "encode_image"
    CATEGORY = "LLM"

    def encode_image(self, image: torch.Tensor):
        img_np = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)
        llm_image = {"type": "image", "image": pil_image}
        return (llm_image,)

class LLMTextGenerator:
    def __init__(self):
        self.loaded_model = None
        self.loaded_processor = None
        self.loaded_tokenizer = None
        self.current_model_name = ""
        self.current_model_type = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (llm_model_list, {"tooltip": "选择要加载的LLM模型文件夹。/ Select the LLM model folder to load."}),
                "model_mode": (["auto", "text", "multimodal"], {"default": "auto", "tooltip": "设置模型加载模式：'auto'自动检测，'text'纯文本，'multimodal'多模态。/ Set model loading mode: 'auto', 'text', or 'multimodal'."}),
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
                "image_1": ("LLM_IMAGE",),
                "image_2": ("LLM_IMAGE",),
                "image_3": ("LLM_IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"

    def generate_text(self, model_name, model_mode, system_prompt, user_prompt, 
                      max_new_tokens, min_new_tokens, do_sample, temperature, top_p, top_k,
                      repetition_penalty, no_repeat_ngram_size, num_beams, length_penalty,
                      image_1=None, image_2=None, image_3=None, prompt=None, extra_pnginfo=None, should_change = False):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        load_as_vision = False
        if model_mode == "auto":
            load_as_vision = "1b" not in model_name
            print(f"模式 'auto': 根据模型名称自动检测。检测结果 -> {'多模态' if load_as_vision else '纯文本'}")
        elif model_mode == "multimodal":
            load_as_vision = True
            print("模式 'multimodal': 强制作为多模态模型加载。")
        else:
            load_as_vision = False
            print("模式 'text': 强制作为纯文本模型加载。")
        
        model_type_changed = (self.current_model_type != ('vision' if load_as_vision else 'text'))
        if self.current_model_name != model_name or self.loaded_model is None or model_type_changed:
            if "模型" in model_name or "读取" in model_name:
                raise Exception(f"无效的模型名称: {model_name}。请检查您的模型文件夹。")

            model_path = os.path.join(llm_base_dir, model_name)
            
            if self.loaded_model is not None:
                del self.loaded_model
                if self.loaded_processor: del self.loaded_processor
                if self.loaded_tokenizer: del self.loaded_tokenizer
                self.loaded_model = self.loaded_processor = self.loaded_tokenizer = None
                torch.cuda.empty_cache()
                print(f"已卸载旧模型: {self.current_model_name}")
            
            print(f"正在从路径加载新模型: {model_path}")
            pbar = comfy.utils.ProgressBar(2)
            try:
                if load_as_vision:
                    self.loaded_processor = AutoProcessor.from_pretrained(model_path)
                    pbar.update(1)
                    self.loaded_model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map="auto").eval()
                    self.current_model_type = 'vision'
                else:
                    self.loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
                    pbar.update(1)
                    self.loaded_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()
                    self.current_model_type = 'text'
                pbar.update(1)
                self.current_model_name = model_name
                print(f"模型 {model_name} 加载成功。")

            except Exception as e:
                self.current_model_name = ""
                self.current_model_type = None
                torch.cuda.empty_cache()
                raise Exception(f"加载模型失败: {e}")
        
        if self.current_model_type == 'vision':
            messages = []
            if system_prompt.strip(): messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            user_content = []
            images = [img for img in [image_1, image_2, image_3] if img is not None]
            if images:
                for img_data in images: user_content.append({"type": "image", "image": img_data['image']})
                print(f"已添加 {len(images)} 张图像到提示中。")
            user_content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": user_content})
            inputs = self.loaded_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.loaded_model.device)
        else:
            if any([image_1, image_2, image_3]): print("警告：当前为纯文本模式，所有图像输入都将被忽略。")
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            inputs = self.loaded_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)

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
            outputs = self.loaded_model.generate(**inputs, **generation_kwargs)
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        decoder = self.loaded_processor if self.current_model_type == 'vision' else self.loaded_tokenizer
        result_text = decoder.decode(generated_tokens[0], skip_special_tokens=True)

        print(f"输出\n{result_text}\n")
        
        return (result_text,)

    @classmethod
    def IS_CHANGED(s, should_change=False, *args, **kwargs):
        if should_change:
            return float("NaN")
        else:
            return False

NODE_CLASS_MAPPINGS = {
    "LLMTextGenerator": LLMTextGenerator,
    "LLMImageEncoder": LLMImageEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTextGenerator": "LLM 文本生成器",
    "LLMImageEncoder": "LLM 图像编码器",
}
