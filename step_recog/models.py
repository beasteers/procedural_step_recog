import torch
from torch import nn


class Decapitvore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.omni = torch.hub.load("facebookresearch/omnivore:main", model=cfg['MODEL']['VID_BACKBONE'])
        self.heads = self.omni.heads
        self.omni.heads = nn.Identity() # decapitate omnivore
        self.embedding_size = cfg['MODEL']['VID_EMBED_SIZE']

    def forward(self, x):
        with torch.no_grad():
            shoul = self.omni(x, input_type="video")
            y_raw = self.heads(shoul)
        return shoul, y_raw

class StepNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.__SKILL_STEPS__ = {k: torch.tensor(v) for k, v in cfg['MODEL']['SKILL_STEPS'].items()}

        self.dropout = nn.Dropout(cfg['MODEL']['GRU_DROPOUT'])

        self.video_backbone = Decapitvore(cfg)
        self.video_dense = nn.Linear(self.video_backbone.embedding_size,cfg['MODEL']['GRU_INPUT_SIZE'])
        self.vid_nframes = cfg['MODEL']['VID_NFRAMES']
        self.register_buffer('vid_mean', torch.tensor(cfg['MODEL']['VID_MEAN']), persistent=False)
        self.register_buffer('vid_std', torch.tensor(cfg['MODEL']['VID_STD']), persistent=False)

        self.gru = nn.GRU(cfg['MODEL']['GRU_INPUT_SIZE'],cfg['MODEL']['GRU_INPUT_SIZE'],cfg['MODEL']['GRU_NUM_LAYERS'],dropout=cfg['MODEL']['GRU_DROPOUT'])
        self.gru_dense_steps = nn.Linear(cfg['MODEL']['GRU_INPUT_SIZE'],cfg['DATASET']['NSTEPS'])
        
        self.use_state_head = cfg['MODEL']['USE_STATE_HEAD']
        if self.use_state_head:
            self.gru_states = nn.GRU(cfg['DATASET']['NSTEPS'],cfg['MODEL']['GRU_INPUT_SIZE'],1,dropout=cfg['MODEL']['GRU_DROPOUT'])
            self.gru_dense_state_machine = nn.Linear(cfg['MODEL']['GRU_INPUT_SIZE'],(cfg['DATASET']['NSTEPS']-1)*cfg['DATASET']['NMACHINESTATES'])

        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(-1)

    def transform_image(self, x):
        if x.ndim == 3:
            # height, width, channel -> batch, time, channel, height, width
            print(x.shape)
            x = x[None, None].permute(0,1,4,2,3)
            print(x.shape)
        x -= self.vid_mean[:, None, None]
        x /= self.vid_std[:, None, None]
        # batch, time, channel, height, width
        # batch, channel, time, height, width
        x = x.permute(0,2,1,3,4)
        return x

    def insert_image_buffer(self, x, output=None):
        if output is None:
            return x.repeat(1, 1, self.vid_nframes, 1, 1)
        output[:-x.shape[2]] = output[x.shape[2]:]
        output[-x.shape[2]:] = x
        return output

    def forward(self, x, h=None):
        h_step, h_state = h if h is not None else (None, None)

        vid_embeds = []
        omni_outs = []
        for t in range(max(x.shape[2]-self.vid_nframes+1, 1)):
            x_vid = x[:,:,t:t+self.vid_nframes]
            assert x_vid.shape[2] == self.vid_nframes
            vid_emb, vid_y_raw = self.video_backbone(x_vid)
            vid_embeds.append(vid_emb)
            omni_outs.append(vid_y_raw)
        vid_embeds = torch.stack(vid_embeds)
        omni_outs = torch.stack(omni_outs)
        vid_embeds = self.dropout(self.relu(self.video_dense(vid_embeds)))
        gru_out, h_step = self.gru(vid_embeds, h_step)
        y_hat_steps = self.gru_dense_steps(gru_out).permute(1,0,2)
        
        y_hat_state_machine = None
        if self.use_state_head:
            gru_out, h_state = self.gru_states(y_hat_steps, h_state)
            y_hat_state_machine = self.gru_dense_state_machine(gru_out).permute(1,0,2)
            y_hat_state_machine = y_hat_state_machine.reshape(y_hat_steps.shape[0],y_hat_steps.shape[1],y_hat_steps.shape[2]-1,-1)

        h = h_step, h_state
        return y_hat_steps, y_hat_state_machine, omni_outs, h

    def skill_step_proba(self, y_hat_steps, y_hat_state_machine, skill=None):
        y_hat_steps_skill = torch.softmax(y_hat_steps[:, :, self.__SKILL_STEPS__[skill]], dim=-1)
        y_hat_state_machine_skill = torch.softmax(y_hat_state_machine[:, :, self.__SKILL_STEPS__[skill]], dim=-1)

        # y_hat_steps = torch.softmax(y_hat_steps, dim=-1)
        # y_hat_steps_skill = y_hat_steps[:, __SKILL_STEPS__[skill]]
        # y_hat_steps_skill[:, -1] = y_hat_steps_skill[:, :-1].sum(1)

        # y_hat_state_machine = torch.softmax(y_hat_state_machine, dim=-1)
        # y_hat_state_machine_skill = y_hat_state_machine[:, __SKILL_STEPS__[skill]]
        # y_hat_state_machine_skill[:, -1] = y_hat_state_machine[:, :-1].sum(1)

        return y_hat_steps_skill, y_hat_state_machine_skill





if __name__ == "__main__":

    config = {
        "MODEL": {
            "VID_BACKBONE": "omnivore_swinB_epic",
        },
        "DATASET": {
        }
    }

    model = Omnivore(config)
    print(model)
