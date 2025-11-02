import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import scipy.io as sio
import json
import os
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .clip_text import clip
from .clip_text.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tqdm
_tokenizer = _Tokenizer()
import numpy as np
import copy
import clip.clip as clip_ori
from utils.mask import MaskingGenerator

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def load_clip_to_cpu_ori():
    backbone_names=['RN50','RN101','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    dims=[1024,1024,512,512,768,768]
    ind=4
    print(backbone_names[ind])
    url = clip_ori._MODELS[backbone_names[ind]]
    model_path = clip_ori._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model,dims[ind]

CUSTOM_TEMPLATES_ori = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of an aircraft {}.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}, a type of car.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": " {}.",
    "OxfordFlowers": " {}.",
    "FGVCAircraft": " {}.",
    "DescribableTextures": " {}.",
    "EuroSAT": " {}.",
    "StanfordCars": " {}.",
    "Food101": " {}.",
    "SUN397": " {}.",
    "Caltech101": " {}.",
    "UCF101": " {}.",
    "ImageNet": "{}.",
    "ImageNetSketch": "{}.",
    "ImageNetV2": "{}.",
    "ImageNetA": "{}.",
    "ImageNetR": "{}.",
}


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, prompt_projections, n_ctx, layers, tokenized_prompts,flag=False):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if flag:
            x = self.transformer(x)
        else:
            counter=0
            outputs = self.transformer.resblocks([x,prompt_projections,n_ctx,layers,counter])
            x = outputs[0]            

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                    *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                    "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX 
        n_ctx_v = cfg.TRAINER.COOP.N_CTX_V 
        ctx_init = cfg.TRAINER.COOP.CTX_INIT #False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            print("use given words to initialize context vectors")
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)     
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC: #False
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context") 
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)#Initial context: "X X X X X X"

        
        print(f'Initial context: "{prompt_prefix}"') #Initial context: "X X X X X X"
        print(f"Number of context words (tokens): {n_ctx}") 

        self.ctx = nn.Parameter(ctx_vectors)  
         
        ctx_vectors_vis = torch.empty(n_ctx_v, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors_vis, std=0.02)
        self.ctx_visual = nn.Parameter(ctx_vectors_vis) 
        
        clip_model_ = load_clip_to_cpu(cfg) 
        clip_model_.cuda()

       
        temp = CUSTOM_TEMPLATES_ori[cfg.DATASET.NAME] 
        classnames = [name.replace("_", " ") for name in classnames] 
        prompts_ = [temp.format(c) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)  


        #------------------------------------
        single_layer = nn.Linear(ctx_dim, 768)  
        single_layer.weight = nn.Parameter(single_layer.weight.to(torch.float16))
        single_layer.bias = nn.Parameter(single_layer.bias.to(torch.float16))              
        self.multi_projections_for_visual = _get_clones(single_layer, 11)
        #------------------------------------


        #-----------------cross_prompts_atten-------------------
        self.atten_net1 = PromptCrossAttention(visual_prompt=768,num_heads=64)
        self.prompt_cross_atten_vision = _get_clones(self.atten_net1, 11)
        convert_weights(self.prompt_cross_atten_vision)

        self.atten_net2 = PromptCrossAttention(visual_prompt=512,num_heads=64)
        self.prompt_cross_atten_text = _get_clones(self.atten_net2, 11)
        convert_weights(self.prompt_cross_atten_text)
        #------------------------------------

     
        classnames = [name.replace("_", " ") for name in classnames] 
        temp_templates = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        temp =prompt_prefix+" "+temp_templates 


        dataset_name = cfg.DATASET.NAME
        # describe_json = osp.join('./simple_describe/deepseek/', dataset_name+'.json')
        describe_json = osp.join('./simple_describe/describe/', dataset_name+'.json')
        with open(describe_json, 'r') as f:
            class_describe = json.load(f)
        prompts = [temp.format(c)+" "+ class_describe[c] for c in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
       
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION #"end" 
        self.prev_ctx=None
        
        self.VPT_fc = nn.Linear(512, self.n_cls, bias=False)

        if cfg.DATASET.SUBSAMPLE_CLASSES=="base" or cfg.DATASET.SUBSAMPLE_CLASSES=="new" :
            self.base_prototype = nn.Parameter(torch.zeros(n_cls, 512), requires_grad=False) 


    def forward(self):
        prefix = self.token_prefix 
        suffix = self.token_suffix 
        ctx = self.ctx 
 
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompt = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
      
        return prompt, self.ctx_visual,self.multi_projections_for_visual,self.prompt_cross_atten_text,self.prompt_cross_atten_vision

class PromptCrossAttention(nn.Module): 
    def __init__(self,visual_prompt=768,num_heads=64):
        super().__init__()
        
        n_head_visual = visual_prompt//num_heads
        self.promt_attention_visual = nn.MultiheadAttention(visual_prompt, n_head_visual)

        self.ln0 = nn.LayerNorm(visual_prompt).to(torch.float16)
        self.mlp_pa_visual = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Linear(visual_prompt,256,bias=False).to(torch.float16)),
            ("gelu1", QuickGELU()),
            ("c_proj", nn.Linear(256, visual_prompt,bias=False).to(torch.float16))
        ]))

        self.scale1 = nn.Parameter(0.6*torch.ones(1))
        self.scale2 = nn.Parameter(0.6*torch.ones(1))
          
    def forward(self, visual_prompts, text_prompts): 
        visual_context = self.scale1.half()*visual_prompts + (1-self.scale1.half())*self.promt_attention_visual(text_prompts,visual_prompts,visual_prompts,need_weights=False )[0]
        visual_context = self.scale2.half()*visual_context + (1-self.scale2.half())*self.mlp_pa_visual(self.ln0(visual_context))


        return visual_context
    


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4,weight=0.1):
        super(Adapter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            QuickGELU(),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )
        self.weight = weight

    def forward(self, x):
        x = self.weight*self.mlp(x)+(1-self.weight)*x
        return x
    

   
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model,device)
        self.classnames=classnames
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ind_ = 0    
        self.n_ctx = cfg.TRAINER.COOP.N_CTX 
        layers=[]
        layer_ =[1,2,3,4,5,6,7,8,9,10,11]

        layers.append(layer_)
        self.layers=layers[cfg.TRAINER.COOP.L_IND] 

        self.adapter_image = Adapter(512, 8, 0.1).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 8, 0.2).to(clip_model.dtype)
        self.mask_image = MaskingGenerator(input_size=224, num_masking_patches=5, min_num_patches=1, max_num_patches=4)
        self.dataset =  cfg.DATASET.NAME
        self.subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        
        if self.subsample == 'all':
            devices=get_mapped_devices()
            self.device = devices[0]
            self.device1 = devices[1]
    
    
    def forward(self, image, label=None,feat_flag=False,proto=None):

        image = self.mask_image(image)    
        text_features_old = self.ori_embedding
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

 
        prompts, visual_ctx, multi_projections_for_visual, prompt_cross_atten_text, prompt_cross_atten_vision = self.prompt_learner()
        with torch.no_grad():
            image_features_fixed = self.image_encoder(image.type(self.dtype))    
        image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)
         
        if feat_flag:
            return image_features_fixed
        
      
        if not self.prompt_learner.training and self.subsample == 'new': 
            image_test_proto =self.prompt_learner.base_prototype.to(torch.float16)
            image_features = self.image_encoder(image.type(self.dtype),visual_ctx,image_test_proto,multi_projections_for_visual,prompt_cross_atten_vision,self.layers)
        else:
            image_proto_info = self.prompt_learner.VPT_fc.weight 
            image_features = self.image_encoder(image.type(self.dtype),visual_ctx,image_proto_info,multi_projections_for_visual,prompt_cross_atten_vision,self.layers)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
        
        image_features = image_features + image_features_fixed
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                             
 
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        if self.subsample == 'all': 
            text_features = self.text_encoder(prompts.to(self.device1), prompt_cross_atten_text.to(self.device1),self.n_ctx,self.layers,tokenized_prompts.to(self.device1).detach()) 
            text_features = text_features.to(self.device)
        else:
            text_features = self.text_encoder(prompts, prompt_cross_atten_text,self.n_ctx,self.layers,tokenized_prompts.detach()) 

        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features + text_features_old
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.adapter_image(image_features)
        text_features = self.adapter_text(text_features)

       
        logits = logit_scale.detach() * image_features @ text_features.t()

        if self.prompt_learner.training:    
            loss = F.cross_entropy(logits, label)
            score = cos(text_features, text_features_old)
            loss_distill_text = 1.0 - torch.mean(score)
            score = cos(image_features, image_features_fixed)
            loss_distill_image = 1.0 - torch.mean(score)
            loss_distill = loss_distill_text*8 + loss_distill_image*6
    
            loss +=loss_distill

            return logits, loss
        else:      
            return logits 
        
def get_mapped_devices():

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    if cuda_visible_devices is not None:

        gpu_ids = cuda_visible_devices.split(",")
        print(f"CUDA_VISIBLE_DEVICES is set to: {gpu_ids}")
        
       
        devices = [torch.device(f"cuda:{i}") for i in range(len(gpu_ids))]
        for i, device in enumerate(devices):
            print(f"Mapped PyTorch device: {device} (Physical GPU ID: {gpu_ids[i]})")
        return devices




@TRAINER_REGISTRY.register()
class CAP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        self.n_cls = len(classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model,self.device)
        self.w = cfg.TRAINER.COOP.W

        print("Turning off gradients in both the image and the text encoder")

        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name and "VPT" not in name:# and "VPT_fc" in name:
                param.requires_grad_(False)



        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None


        device_count = torch.cuda.device_count()
        if cfg.DATASET.SUBSAMPLE_CLASSES == "all" : #cfg.DATASET.NAME == 'ImageNet' and  
            devices = get_mapped_devices()   
            if len(devices) == 2: 
                device_1 = devices[0]  
                device_2 = devices[1] 
            else:
                print("Less than two devices are available.")
            self.device = device_1
            device1 = device_2
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
        elif device_count > 1:
            self.model = nn.DataParallel(self.model)
        else:
            self.model.to(self.device)

        self.proto = self.get_prototype()
        self.model.prompt_learner.VPT_fc.weight.data=self.proto

        if cfg.DATASET.SUBSAMPLE_CLASSES=="base":
             with torch.no_grad():
                self.model.prompt_learner.base_prototype.copy_(self.proto)

            
    
    def get_prototype(self):
        self.set_model_mode("eval")
        data_loader = self.train_loader_x
        embedding_list=[]
        label_list=[]
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            with torch.no_grad():
                image_feature = self.model(input,feat_flag=True)
            embedding_list.append(image_feature)
            label_list.append(label)
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_num = len(torch.unique(label_list))
        proto_list = []
        for class_index in range(class_num):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]

            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto = torch.stack(proto_list, dim=0)
 
        proto = proto / proto.norm(dim=-1, keepdim=True)
        return proto


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC #fp16
        if prec == "amp": 
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,loss = self.model(image, label,proto=self.proto)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        # print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
              
            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            if "VPT_fc.weight" in state_dict:
                del state_dict["VPT_fc.weight"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            if hasattr(self._models.get(name), 'base_prototype'):
                with torch.no_grad():
                    old_base_prototype =  state_dict['base_prototype']
                    if self.model.prompt_learner.base_prototype.size(0) != old_base_prototype.size(0): # now >old
                        expand_size = old_base_prototype.size(0) - self.model.prompt_learner.base_prototype.size(0)
                        expanded_base_prototype = torch.cat([self.model.prompt_learner.base_prototype, 
                                                             torch.ones(expand_size, old_base_prototype.size(1)).to(self.model.prompt_learner.base_prototype.device)], dim=0)
                        self.model.prompt_learner.base_prototype.data = expanded_base_prototype
                        self.model.prompt_learner.base_prototype.data[:old_base_prototype.size(0), :] = old_base_prototype
                state_dict.pop('base_prototype', None)
     
            self._models[name].load_state_dict(state_dict, strict=False)
