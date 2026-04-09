import cv2
import os
import numpy as np
from torch import tensor
import torch

def convertYolo2xyxy(yololabel, img_w, img_h):
  w = yololabel[3]*img_w
  h = yololabel[4]*img_h
  minX = ((2 * yololabel[1] * img_w) - w)/2
  minY = ((2 * yololabel[2] * img_h) - h)/2
  maxX = minX + w
  maxY= minY + h
  return(minX, minY, maxX, maxY)

def format_preds_mAP(box, score, label=None):
  
  dic = {'boxes':[], 'scores': [], 'labels': []}
  if label is None:
    if len(box) == 0:
      label = []
    else:
      label = [0] * len(box[0])

  if isinstance(box, torch.Tensor):
    if len(box) > 0:
      for i in range(len(box)):
        dic['boxes'].append(box[i])
        dic['scores'].append(score[0][i])
        dic['labels'].append(label[0][i])
  else:
    if len(box[0]) > 0:
      for i in range(len(box)):
        dic['boxes'].append(box[0][i])
        dic['scores'].append(score[0][i])
        dic['labels'].append(label[0][i])

  dic['boxes'] = tensor(np.array(dic['boxes']))
  dic['scores'] = tensor(np.array(dic['scores']))
  dic['labels'] = tensor(np.array(dic['labels']))
  return dic

def format_labels_mAP(box, label=None):
  if label is None:
    label = [0] * len(box)
  dic = {'boxes':[], 'labels': []}
  for i in range(len(box)):
    dic['boxes'].append(box[i])
    dic['labels'].append(label[i])
     
  dic['boxes'] = tensor(np.array(dic['boxes']))
  dic['labels'] = tensor(np.array(dic['labels']))
  return dic

def save_bbox_annotations(fullimgpath, groundTruth_bboxes, pred_bboxes, outdir):
  os.makedirs(outdir, exist_ok=True)
  outfile = os.path.join(outdir, os.path.basename(fullimgpath))
  dispimage = cv2.imread(fullimgpath)
  for groundtruth in groundTruth_bboxes:
    cv2.rectangle(dispimage, (int(groundtruth[0]), int(groundtruth[1])), (int(groundtruth[2]), int(groundtruth[3])), color=(0,255,255), thickness=3)
  if len(pred_bboxes) > 0:
    for preds in pred_bboxes:
      cv2.rectangle(dispimage, (int(preds[0]), int(preds[1])), (int(preds[2]), int(preds[3])), color=(0,0,255), thickness=3)
  cv2.imwrite(outfile, dispimage)
  
def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt

    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer
