import argparse
from collections import defaultdict

import numpy as np
import torch
import os
import json
import pandas as pd
from PIL import Image
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path, KeywordsStoppingCriteria

from llava.eval.utils import visualize_token_to_vis_token_attn_scores

import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

OBJECT_TOKEN_INDEX = -300
def tokenizer_image_object_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, object_token_index=OBJECT_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = []
    for prompt_chunk in prompt.split('<image>'):
        prompt_chunks.extend(prompt_chunk.split('<object>'))
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
    def insert_separator(X, seps):
        return [ele for sublist in zip(X, seps) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    sep = [[image_token_index] * (offset + 1)] + [[object_token_index] * (offset + 1)]*(len(prompt_chunks)-1)
    for x in insert_separator(prompt_chunks, sep):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


@torch.inference_mode()
def multiple_choices_inference(model, tokenizer, image_processor, conv_type, image, question, options, object_crops=None, images_long=None, objects_long=None, do_attn_analysis=False):
    conv = conv_templates[conv_type].copy()
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    question_input_ids = tokenizer_image_object_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                                      return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    output_question = model(
        question_input_ids,
        use_cache=True,
        images=image_tensor.unsqueeze(0).half().cuda(),
        # object_features=object_crops.half().cuda() if object_crops is not None else None,
        # images_long=images_long,
        # objects_long=objects_long
    )

    question_logits = output_question.logits
    question_past_key_values = output_question.past_key_values

    loss_list = []

    for option in options:
        conv = conv_templates[conv_type].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], option)
        full_prompt = conv.get_prompt()

        full_input_ids = tokenizer_image_object_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                                      return_tensors='pt').unsqueeze(0).cuda()
        option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

        output_option = model(input_ids=option_answer_input_ids,
                              use_cache=True,
                              attention_mask=torch.ones(1, question_logits.shape[1] + option_answer_input_ids.shape[1],
                                                        device=full_input_ids.device),
                              past_key_values=question_past_key_values)

        logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, model.config.vocab_size)
        labels = option_answer_input_ids.view(-1)
        loss = loss_fct(logits, labels)

        loss_list.append(loss)

    option_chosen = torch.stack(loss_list).argmin()

    if do_attn_analysis:
        outputs_for_attn_analysis = model(question_input_ids,
                                          use_cache=True,
                                          images=image_tensor.unsqueeze(0).half().cuda(),
                                          # object_features=object_crops.half().cuda() if object_crops is not None else None,
                                          # images_long=images_long,
                                          # objects_long=objects_long,
                                          output_attentions=True
                                          )

        batch_size = question_input_ids.size(0)
        assert batch_size == 1
        last_token_attn_scores = outputs_for_attn_analysis.attentions[-1][0, :, -1, :]
        avg_last_token_attn_scores = torch.mean(last_token_attn_scores, dim=0)
        all_image_token_indices = outputs_for_attn_analysis.all_image_token_indices[0]
        last_token_to_all_image_token_attn_scores = avg_last_token_attn_scores[all_image_token_indices].cpu().tolist()
        CLS_tok_image_attentions = outputs_for_attn_analysis.image_attentions[0].cpu().tolist()
    else:
        last_token_to_all_image_token_attn_scores = None
        CLS_tok_image_attentions = None

    return option_chosen.cpu().item(), last_token_to_all_image_token_attn_scores, CLS_tok_image_attentions


def expand2square(pil_img, background_color):
	width, height = pil_img.size
	if width == height:
		return pil_img, 0, 0
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))
		return result, 0, (width - height) // 2
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))
		return result, (height - width) // 2, 0


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.attn_implementation)
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    results = {}
    per_type_acc = defaultdict(list)
    all_acc = []

    missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
    focus_msg = "Additional visual information to focus on: "
    for test_type in ['direct_attributes', 'relative_position']:
        all_last_token_to_all_image_token_attn_scores = []
        all_CLS_tok_image_attentions = []

        results[test_type] = []
        folder = os.path.join(args.image_folder, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        progress = tqdm(image_files)

        for image_file in progress:
            result_single_sample = {}
            image_path = os.path.join(folder, image_file)
            # Split from the end
            annotation_path = ".".join(image_path.split('.')[:-1] + ['json'])
            image = Image.open(image_path).convert('RGB')
            annotation = json.load(open(annotation_path))
            # image, _, _ = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image, _, _ = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))

            question = annotation['question']
            # generate free-form response to check whether visual search needs to be activated
            # prediction = vqa_llm.free_form_inference(image, question)
            # missing_objects = []
            # if missing_objects_msg in prediction:
            #     missing_objects = prediction.split(missing_objects_msg)[-1]
            #     if missing_objects.endswith('.'):
            #         missing_objects = missing_objects[:-1]
            #     missing_objects = missing_objects.split(',')
            #     missing_objects = [missing_object.strip() for missing_object in missing_objects]
            #
            # search_result = []
            # if len(missing_objects) > 0:
            #     # visual search
            #     for object_name in missing_objects:
            #         image = Image.open(image_path).convert('RGB')
            #         smallest_size = max(int(np.ceil(min(image.width, image.height) / args.minimum_size_scale)),
            #                             args.minimum_size)
            #         final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm, image, object_name,
            #                                                                                     target_bbox=None,
            #                                                                                     smallest_size=smallest_size)
            #         if all_valid_boxes is not None:
            #             # might exist multiple target instances
            #             for search_bbox in all_valid_boxes:
            #                 search_final_patch = final_step['bbox']
            #                 search_bbox[0] += search_final_patch[0]
            #                 search_bbox[1] += search_final_patch[1]
            #                 search_result.append({'bbox': search_bbox.tolist(), 'name': object_name})
            #         else:
            #             search_bbox = final_step['detection_result']
            #             search_final_patch = final_step['bbox']
            #             search_bbox[0] += search_final_patch[0]
            #             search_bbox[1] += search_final_patch[1]
            #             search_result.append({'bbox': search_bbox.tolist(), 'name': object_name})
            # predict the multiple-choice option
            options = annotation['options']
            image = Image.open(image_path).convert('RGB')
            # if len(missing_objects) > 0:
            #     object_names = [_['name'] for _ in search_result]
            #     bboxs = deepcopy([_['bbox'] for _ in search_result])
            #     if len(object_names) <= 2:
            #         images_long = [False]
            #         objects_long = [True] * len(object_names)
            #     else:
            #         images_long = [False]
            #         objects_long = [False] * len(object_names)
            #     object_crops = []
            #     for bbox in bboxs:
            #         object_crop = vqa_llm.get_object_crop(image, bbox, patch_scale=1.2)
            #         object_crops.append(object_crop)
            #     object_crops = torch.stack(object_crops, 0)
            #     image, left, top = expand2square(image, tuple(int(x * 255) for x in vqa_llm.image_processor.image_mean))
            #     bbox_list = []
            #     for bbox in bboxs:
            #         bbox[0] += left
            #         bbox[1] += top
            #         bbox_list.append(bbox)
            #     bbox_list = [normalize_bbox(bbox, image.width, image.height) for bbox in bbox_list]
            #     cur_focus_msg = focus_msg
            #     for i, (object_name, bbox) in enumerate(zip(object_names, bbox_list)):
            #         cur_focus_msg = cur_focus_msg + "{} <object> at location [{:.3f},{:.3f},{:.3f},{:.3f}]".format(
            #             object_name, bbox[0], bbox[1], bbox[2], bbox[3])
            #         if i != len(bbox_list) - 1:
            #             cur_focus_msg = cur_focus_msg + "; "
            #         else:
            #             cur_focus_msg = cur_focus_msg + '.'
            #     question_with_focus = cur_focus_msg + "\n" + question
            #     option_chosen = vqa_llm.multiple_choices_inference(image, question_with_focus, options, object_crops,
            #                                                        images_long=images_long, objects_long=objects_long)
            # else:
            option_chosen, last_token_to_all_image_token_attn_scores, CLS_tok_image_attentions = multiple_choices_inference(model, tokenizer, image_processor, args.conv_mode, image, question, options, do_attn_analysis=True)

            # Note: the first option of all options is always the correct one
            # which is why we consider it's correct if option_chosen == 0
            correct = 1 if option_chosen == 0 else 0
            per_type_acc[test_type].append(correct)
            all_acc.append(correct)

            # if correct == 0:
            #     print(option_chosen, question, options)
            #     breakpoint()

            result_single_sample['question'] = question
            result_single_sample['options'] = options
            result_single_sample['image'] = image_file
            # result_single_sample['prediction_freeform'] = prediction
            # result_single_sample['missing_objects'] = missing_objects
            # result_single_sample['search_result'] = search_result
            result_single_sample['option_chosen'] = option_chosen
            result_single_sample['correct'] = correct
            result_single_sample['last_token_to_all_image_token_attn_scores'] = last_token_to_all_image_token_attn_scores
            result_single_sample['CLS_tok_image_attentions'] = CLS_tok_image_attentions
            all_last_token_to_all_image_token_attn_scores.append(last_token_to_all_image_token_attn_scores)
            all_CLS_tok_image_attentions.append(CLS_tok_image_attentions)

            results[test_type].append(result_single_sample)

            progress.set_description(f"{test_type} Acc: {np.mean(per_type_acc[test_type]) * 100:.1f}%")

        with open(os.path.join(args.output_dir, f"answers_{test_type}.json"), 'w') as f:
            json.dump(results[test_type], f, indent=4)
        print(test_type, np.mean(per_type_acc[test_type]))
        acc_str = f"{test_type} Accuracy: {np.mean(per_type_acc[test_type]) * 100:.1f}% ({sum(per_type_acc[test_type])}/{len(per_type_acc[test_type])})"
        with open(os.path.join(args.output_dir, "accuracy.txt"), "a") as f:
            f.write(acc_str + "\n")

        # Visualize avg all_last_token_to_all_image_token_attn_scores and all_CLS_tok_image_attentions
        avg_last_token_to_all_image_token_attn_scores = np.average(np.array(all_last_token_to_all_image_token_attn_scores), axis=0)
        visualize_token_to_vis_token_attn_scores(avg_last_token_to_all_image_token_attn_scores, "Last Text To Image Token Attn Score", os.path.join(args.output_dir, f"{test_type}_last_txt_to_image_attn_score.png"))
        avg_CLS_tok_image_attentions = np.average(np.array(all_CLS_tok_image_attentions), axis=0)
        visualize_token_to_vis_token_attn_scores(avg_CLS_tok_image_attentions, "CLS To Image Token Attn Score", os.path.join(args.output_dir, f"{test_type}_CLS_image_attn_score.png"))

    print(np.mean(all_acc))
    acc_str = f"Overall Accuracy: {np.mean(all_acc) * 100:.1f}% ({sum(all_acc)}/{len(all_acc)})"
    with open(os.path.join(args.output_dir, "accuracy.txt"), "a") as f:
        f.write(acc_str + "\n")
    print(f"Finished evaluating {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")

    parser.add_argument("--attn_implementation", type=str, default="eager")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_model(args)
