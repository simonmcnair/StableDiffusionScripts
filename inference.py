
import clip_interrogator
from clip_interrogator import Config, Interrogator, list_clip_models
from transformers import BlipProcessor, BlipForConditionalGeneration
import utils
from PIL import PngImagePlugin, Image
import pandas as pd
import cv2
import numpy as np
from onnxruntime import InferenceSession
from huggingface_hub import hf_hub_download
from typing import Mapping, Tuple, Dict
import re

logger = utils.get_logger(__name__ + '.log',__name__ + '_error.log')

RE_SPECIAL = re.compile(r'([\\()])')


@utils.timing_decorator
def ddb(imagefile):

    #os.environ["CUDA_VISIBLE_DEVICES"]=""

    #blip_large
    import torch
    from torchvision import transforms
    import json
    import urllib, urllib.request

    #Blip_large
    from transformers import BlipProcessor, BlipForConditionalGeneration


    # Load the model

    model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
    model.eval()


    input_image = Image.open(imagefile) # load an image of your choice
    preprocess = transforms.Compose([
        transforms.Resize(360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
        class_names = json.loads(url.read().decode())

    # The output has unnormalized scores. To get probabilities, you can run a sigmoid on it.
    probs =  torch.sigmoid(output[0]) # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags
    thresh=0.2
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = 'Predictions with probabilities above ' + str(thresh) + ':\n'
    for i in inds[0:len(tmp)]:
        txt += class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())
    #plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt)
    return txt

@utils.timing_decorator
def unumcloud(image):

    #unumcloud(image):
    from transformers import AutoModel, AutoProcessor
    import torch
    model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

    #prompt = "Question or Instruction"
    prompt = ""
    image = Image.open(image)

    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]


@utils.timing_decorator
def nlpconnect(fileinput):
    #nlpconnect
    from transformers import pipeline

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return image_to_text(fileinput)

@utils.timing_decorator
def blip2_opt_2_7b(inputfile):
    #blip2_opt_2_7b
    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from transformers import AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    # pip install accelerate bitsandbytes
    # pip install -q -U bitsandbytes
    # pip install -q -U git+https://github.com/huggingface/transformers.git
    # pip install -q -U git+https://github.com/huggingface/peft.git
    # pip install -q -U git+https://github.com/huggingface/accelerate.git

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    #ViT-L-14/openai

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    #model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_4bit=True, device_map="auto")
    model_nf4 = AutoModelForCausalLM.from_pretrained("Salesforce/blip2-opt-2.7b", quantization_config=nf4_config)

    raw_image = Image.open(inputfile).convert('RGB')

    #question = "how many dogs are in the picture?"
    question = ""
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    logger.info(processor.decode(out[0], skip_special_tokens=True).strip())
    return out[0]

@utils.timing_decorator
def blip_large(imagepath,model='small'):

    if model == 'small':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    elif model == 'large':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    raw_image = Image.open(imagepath).convert('RGB')

    # conditional image captioning
    text = ""
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    logger.info(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)

    res = processor.decode(out[0], skip_special_tokens=True)
    return res

@utils.timing_decorator
def image_make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

@utils.timing_decorator
def image_smart_resize(img, size):
    # Assumes the image has already gone through image_make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    else:  # just do nothing
        pass

    return img

class CLIPInterrogator:
    def __init__(
            self,
            repo='SmilingWolf/wd-v1-4-vit-tagger-v2',
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            mode: str = "auto"
    ) -> None:
        self.__repo = repo
        self.__model_path = model_path
        self.__tags_path = tags_path
        self._provider_mode = mode

        self.__initialized = False
        self._model, self._tags = None, None

    def _init(self) -> None:
        if self.__initialized:
            return

        model_path = hf_hub_download(self.__repo, filename=self.__model_path)
        tags_path = hf_hub_download(self.__repo, filename=self.__tags_path)
        logger.info(f"model path is {model_path}")

        self._model = InferenceSession(str(model_path))
        self._tags = pd.read_csv(tags_path)

        self.__initialized = True

    def _calculation(self, image: Image.Image)  -> pd.DataFrame:
        self._init()

        _, height, _, _ = self._model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = image_make_square(image, height)
        image = image_smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self._model.get_inputs()[0].name
        label_name = self._model.get_outputs()[0].name
        confidence = self._model.run([label_name], {input_name: image})[0]

        full_tags = self._tags[['name', 'category']].copy()
        full_tags['confidence'] = confidence[0]

        return full_tags

    def interrogate(self, image: Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        full_tags = self._calculation(image)

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(full_tags[full_tags['category'] == 9][['name', 'confidence']].values)

        # rest are regular tags
        tags = dict(full_tags[full_tags['category'] != 9][['name', 'confidence']].values)

        return ratings, tags

CLIPInterrogatorModels: Mapping[str, CLIPInterrogator] = {
    'wd14-vit-v2': CLIPInterrogator(),
    'wd14-convnext': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-convnext-tagger'),
    'ViT-L-14/openai': CLIPInterrogator(),
    'ViT-H-14/laion2b_s32b_b79': CLIPInterrogator(),
    'ViT-L-14/openai': CLIPInterrogator(),
    'wd-v1-4-moat-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-moat-tagger-v2'),
    'wd-v1-4-swinv2-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-swinv2-tagger-v2'),
    'wd-v1-4-convnext-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-convnext-tagger-v2'),
    'wd-v1-4-convnextv2-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-convnextv2-tagger-v2'),
    'wd-v1-4-vit-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-vit-tagger-v2')
}

@utils.timing_decorator
def image_to_wd14_tags(filename,modeltouse='wd14-vit-v2') \
        -> Tuple[Mapping[str, float], str, Mapping[str, float]]:
    
    try:
        image = Image.open(filename)
        logger.info("image: " + filename + " successfully opened.  Continue processing ")
    except Exception as e:
        logger.error("Processfile Exception1: " + " failed to open image : " + filename + ". FAILED Error: " + str(e) + ".  Skipping")
        return None

    try:
        logger.info(modeltouse)
        model = CLIPInterrogatorModels[modeltouse]
        ratings, tags = model.interrogate(image)

        filtered_tags = {
            tag: score for tag, score in tags.items()
            #if score >= .35
            if score >= .80
        }

        text_items = []
        tags_pairs = filtered_tags.items()
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
        for tag, score in tags_pairs:
            tag_outformat = tag
            tag_outformat = tag_outformat.replace('_', ' ')
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
            text_items.append(tag_outformat)
        #output_text = ', '.join(text_items)
        #return ratings, output_text, filtered_tags
        return ratings, text_items, filtered_tags
    except Exception as e:
        logger.error(f"Exception getting tags from image {filename}.  Error: {e}" )
        return None

