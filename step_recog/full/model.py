import numpy as np
import torch
from torch import nn
from collections import deque
from ultralytics import YOLO

from act_recog.models import Omnivore
from act_recog.config import load_config as act_load_config

from step_recog.config import load_config
from step_recog.models import OmniGRU

from step_recog.full.clip_patches import ClipPatches
from step_recog.full.download import cached_download_file



class StepPredictor(nn.Module):
    """Step prediction model that takes in frames and outputs step probabilities.
    """
    def __init__(self, cfg_file):
        super().__init__()
        # load config
        self.cfg = load_config(cfg_file)
        self.omni_cfg = act_load_config(self.cfg.MODEL.OMNIVORE_CONFIG)

        # assign vocabulary
        self.STEPS = np.array([
            step
            for skill in self.cfg.SKILLS
            for step in skill['STEPS']
        ])
        self.STEP_SKILL = np.array([
            skill['NAME']
            for skill in self.cfg.SKILLS
            for _ in skill['STEPS']
        ])
        self.MAX_OBJECTS = 25
        
        # build model
        self.head = OmniGRU(self.cfg, load=True)
        if self.head.use_action:
            self.omnivore = Omnivore(self.omni_cfg)
        if self.head.use_objects:
            yolo_checkpoint = cached_download_file(self.cfg.MODEL.YOLO_CHECKPOINT_URL)
            self.yolo = YOLO(yolo_checkpoint)
            self.yolo.eval = lambda *a: None
            self.clip_patches = ClipPatches()
        if self.head.use_audio:
            raise NotImplementedError("Audio is not supported atm")

        # frame buffers and model state
        self.omnivore_input_queue = deque(maxlen=self.omni_cfg.DATASET.FPS * self.omni_cfg.MODEL.WIN_LENGTH)
        self.h = None  

    def reset(self):
        self.omnivore_input_queue.clear()
        self.h = None

    def queue_frame(self, image):
      X_omnivore = self.omnivore.prepare_image(image)
      self.omnivore_input_queue.append(X_omnivore)

    def has_omni_maxlen(self):
      return len(self.omnivore_input_queue) == self.omnivore_input_queue.maxlen  

    def forward(self, image, return_objects=False):
#        pdb.set_trace()
        # compute yolo
        Z_objects = Z_frame = None
        if self.head.use_objects:
            obj_results = self.yolo(image, verbose=False)
            boxes = obj_results[0].boxes
            Z_clip = self.clip_patches(image, boxes.xywh.cpu().numpy(), include_frame=True)

            # concatenate with boxes and confidence
            Z_frame = torch.cat([Z_clip[:1], torch.tensor([[0, 0, 1, 1, 1]]).to(Z_clip.device)], dim=1)
            Z_objects = torch.cat([Z_clip[1:], boxes.xyxyn, boxes.conf[:, None]], dim=1)  ##deticn_bbn.py:Extractor.compute_store_clip_boxes returns xyxyn
            # pad boxes to size
            _pad = torch.zeros((max(self.MAX_OBJECTS - Z_objects.shape[0], 0), Z_objects.shape[1])).to(Z_objects.device)
            Z_objects = torch.cat([Z_objects, _pad])[:self.MAX_OBJECTS]
            Z_frame = Z_frame[None].float()
            Z_objects = Z_objects[None,None].float()

        # compute audio embeddings
        Z_audio = None
        if self.head.use_audio:
            raise NotImplementedError("Audio embeddings not implemented rn")

        # compute video embeddings
        Z_action = None
        if self.head.use_action:
            # rolling buffer of omnivore input frames
            self.queue_frame(image)

            # compute omnivore embeddings
            X_omnivore = torch.stack(list(self.omnivore_input_queue), dim=1)[None]
            frame_idx = np.linspace(0, self.omnivore_input_queue.maxlen - 1, self.omni_cfg.MODEL.NFRAMES).astype('long') #same as act_recog.dataset.milly.py:pack_frames_to_video_clip
            X_omnivore = X_omnivore[:, :, frame_idx, :, :]
            _, Z_action = self.omnivore(X_omnivore.to(Z_objects.device), return_embedding=True)
            Z_action = Z_action[None]

        # mix it all together
        self.h = self.head.init_hidden(Z_action.shape[0])
        prob_step, self.h = self.head(Z_action, self.h, Z_audio, Z_objects, Z_frame)
        prob_step = torch.softmax(prob_step[..., :-2], dim=-1) #prob_step has <n classe positions> <1 no step position> <2 begin-end frame identifiers>
        
        if return_objects:
            return prob_step, obj_results
        return prob_step
