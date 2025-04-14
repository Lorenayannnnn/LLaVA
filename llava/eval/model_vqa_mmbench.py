import argparse

import numpy as np
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

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


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.attn_implementation)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    is_correct = 0
    total_cnt = 0
    all_last_token_to_all_image_token_attn_scores = []
    all_CLS_tok_image_attentions = []
    progress = tqdm(questions.iterrows(), total=len(questions))
    for index, row in progress:
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            raise NotImplementedError("All rounds not supported yet")
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            answer = row['answer']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

                outputs_for_attn_analysis = model(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), output_attentions=True)

            batch_size = input_ids.size(0)
            assert batch_size == 1
            last_token_attn_scores = outputs_for_attn_analysis.attentions[-1][0, :, -1, :]
            avg_last_token_attn_scores = torch.mean(last_token_attn_scores, dim=0)
            all_image_token_indices = outputs_for_attn_analysis.all_image_token_indices[0]
            last_token_to_all_image_token_attn_scores = avg_last_token_attn_scores[all_image_token_indices].cpu().tolist()
            CLS_tok_image_attentions = outputs_for_attn_analysis.image_attentions[0].mean(0).cpu().tolist()

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {},
                                    "gt_answer": answer,
                                    "last_token_to_vis_token_attn_scores": last_token_to_all_image_token_attn_scores,
                                   "image_attentions": CLS_tok_image_attentions
                                       }) + "\n")
            ans_file.flush()

            all_last_token_to_all_image_token_attn_scores.append(last_token_to_all_image_token_attn_scores)
            all_CLS_tok_image_attentions.append(CLS_tok_image_attentions)

            if outputs.startswith(answer):
                is_correct += 1
            else:
                pass
                # print(f"Output: {outputs}")
                # print(f"Answer: {answer}")
                # import pdb
                # pdb.set_trace()
            total_cnt += 1
            progress.set_description(f"Acc: {is_correct / total_cnt * 100:.1f}%")

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

    # Output accuracy
    acc_str = f"Accuracy: {is_correct / total_cnt * 100:.1f}% ({is_correct}/{total_cnt})"
    print(acc_str)
    output_dir = os.path.dirname(args.answers_file)
    with open(os.path.join(output_dir, "accuracy.txt"), "w") as f:
        f.write(acc_str + "\n")

    # Visualize avg all_last_token_to_all_image_token_attn_scores and all_CLS_tok_image_attentions
    avg_last_token_to_all_image_token_attn_scores = np.average(np.array(all_last_token_to_all_image_token_attn_scores), axis=0)
    visualize_token_to_vis_token_attn_scores(avg_last_token_to_all_image_token_attn_scores, "Last Text To Image Token Attn Score", os.path.join(output_dir, "last_txt_to_image_attn_score.png"))
    avg_CLS_tok_image_attentions = np.average(np.array(all_CLS_tok_image_attentions), axis=0)
    visualize_token_to_vis_token_attn_scores(avg_CLS_tok_image_attentions, "CLS To Image Token Attn Score", os.path.join(output_dir, "CLS_image_attn_score.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
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

    eval_model(args)
