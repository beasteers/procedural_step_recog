import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import tqdm
import yaml
import cv2
import torch
import supervision as sv
import numpy as np
from step_recog.models import StepNet
from step_recog.statemachine import ProcedureStateMachine

STATES = [
    (128, 128, 128), ##unobserved = grey
    (217, 217, 38),  ##current = blue
    (38, 217, 38)    ##done = green
]

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device='cpu'

import ipdb
@ipdb.iex
@torch.no_grad()
def main(video_path, skill, checkpoint_path, output_path='output.mp4', fps=10, cfg_file="step_recog/config.yaml"):
    '''Visualize the outputs of the model on a video.

    '''
    # create video reader and video writer
    video_info = sv.VideoInfo.from_video_path(video_path)

    # define model
    cfg = yaml.load(open(cfg_file), Loader=yaml.CLoader)
    model = StepNet(cfg).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    sm = ProcedureStateMachine(len(model.__SKILL_STEPS__[skill]))

    h = None
    buffer = None
    with sv.VideoSink(output_path, video_info=video_info) as sink:
        # iterate over video frames
        for idx, frame in tqdm.tqdm(enumerate(sv.get_video_frames_generator(video_path))):
            if idx % int(video_info.fps / fps) != 0:
               continue

            x = torch.tensor(frame.astype(np.float32), device=device)
            x = model.transform_image(x)
            buffer = model.insert_image_buffer(x, buffer)
            y_hat_steps, y_hat_state_machine, omni_outs, h = model(buffer, h)
            prob_step, p_state = model.skill_step_proba(y_hat_steps, y_hat_state_machine, skill=skill)
            prob_step = prob_step[0, -1]
            if model.use_state_head:
                p_state = p_state[0, -1]
                idx_state = torch.argmax(p_state, dim=-1)
            else:
                sm.process_timestep(prob_step)
                idx_state = sm.current_state
            
            step_idx  = np.argmax(y_hat_steps, dim=-1)
            step_desc = "No step" if step_idx >= len(model.STEPS) else model.STEPS[step_idx]              

            # draw the prediction (could be your bar chart) on the frame
            plot_graph(frame, prob_step, step_desc, idx_state)
            sink.write_frame(frame)

    ##TODO: Review the offsets
    ##colors in BGR
def plot_graph(frame, prob_step, step_desc, current_state, tl=(10, 25), scale=1.0, bar_space=10, text_color=(0, 219, 219), bar_clor=(22, 22, 197), thickness=1):
    width       = 30
    height      = 100
    start_point = (50, 50)
    border_color = (0, 0, 0)
    max_desc_length = 62
    end_point   = (start_point[0] + width, start_point[1] + height)

    cv2.putText(frame, "Step: " + step_desc[:max_desc_length], (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_COMPLEX, scale, border_color, thickness * 2) #black border
    cv2.putText(frame, "Step: " + step_desc[:max_desc_length], (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_COMPLEX, scale, text_color, thickness)

    if prob_step is not None:
        for i, prob in enumerate(prob_step):
            current_start = (start_point[0] + i * (width + bar_space), int( height * (1 - prob) + start_point[1]) )
            current_end   = (current_start[0] + width, end_point[1])  

            cv2.rectangle(frame, tuple(a - 1 for a in current_start), tuple(a + 1 for a in current_end), border_color, -1) #black border        
            cv2.rectangle(frame, current_start, current_end, bar_clor, -1)     

            if i == len(prob_step) - 1:
                cv2.line(frame, (start_point[0], end_point[1] + bar_space), (current_end[0], end_point[1] + bar_space), border_color, 2) #black border                
                cv2.line(frame, (start_point[0], end_point[1] + bar_space), (current_end[0], end_point[1] + bar_space), text_color, 1)        

            cv2.putText(frame, str(i + 1), (current_start[0] + int(width / 2),  end_point[1] + tl[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border  
            cv2.putText(frame, str(i + 1), (current_start[0] + int(width / 2),  end_point[1] + tl[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2)) 

        ## ===================================================================================================================================================================# 
            if i < len(current_state):
                state_current_start = (current_start[0], end_point[1] + 40)
                state_current_end   = (current_end[0], end_point[1] + 55)

                cv2.rectangle(frame, tuple(a - 1 for a in state_current_start), tuple(a + 1 for a in state_current_end), border_color, -1) #black border 
                cv2.rectangle(frame, state_current_start, state_current_end, STATES[current_state[i]], -1)     
        ## ===================================================================================================================================================================#            

        cv2.line(frame, (start_point[0] - bar_space + 2, end_point[1]), (start_point[0] - bar_space + 2, start_point[1]), border_color, 2)
        cv2.line(frame, (start_point[0] - bar_space + 2, end_point[1]), (start_point[0] - bar_space + 2, start_point[1]), text_color, 1)      

        cv2.putText(frame, "100", (start_point[0] - 40,  start_point[1] + 10), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border 
        cv2.putText(frame, "100", (start_point[0] - 40,  start_point[1] + 10), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2)) 

        cv2.putText(frame, "0",   (start_point[0] - 20,  end_point[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border        
        cv2.putText(frame, "0",   (start_point[0] - 20,  end_point[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2))   


if __name__ == '__main__':
    import fire
    fire.Fire(main)
