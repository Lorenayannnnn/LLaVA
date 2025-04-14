import argparse
import re
from ast import literal_eval
from collections import defaultdict

import numpy as np
from datasets import load_dataset
import torch
import os
import json
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images

from llava.eval.utils import visualize_token_to_vis_token_attn_scores


all_options = ['A', 'B', 'C', 'D']
prompt_templates = {
    "vision": "Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    "standard": "Answer with the option letter from the given choices directly."
}


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


def replace_images_tokens(input_string):
    image_order = [int(num) for num in re.findall(r'<image\s+(\d+)>', input_string)]
    # input_string = re.sub(r'<image\s+\d+>', '<image>', input_string)
    input_string = re.sub(r'<image\s+\d+>', 'image', input_string)
    return input_string, image_order


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{prompt_templates['standard']}"
    return question


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)


def origin_mmmu_doc_to_visual(doc, image_order):
    visual = []
    for idx in image_order:
        visual.append(doc[f'image_{idx}'])
    return visual


def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]


def process_prompt(data, test_type):
    if 'standard' in test_type:
        prompt, image_order = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data, image_order)
    elif test_type == 'vision':
        prompt = prompt_templates['vision']
        images = vision_mmmu_doc_to_visual(data)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    return (prompt, images)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.attn_implementation)

    results = {}
    per_type_acc = defaultdict(list)
    all_acc = []
    all_last_token_to_all_image_token_attn_scores = []
    all_CLS_tok_image_attentions = []

    # for test_type in ["vision", "standard (4 options)", "standard (10 options)"]:
    for test_type in ["standard (4 options)", "standard (10 options)"]:
        dataset = load_dataset("MMMU/MMMU_Pro", test_type)["test"]
        test_type_last_token_to_all_image_token_attn_scores = []
        test_type_CLS_tok_image_attentions = []

        # Skip multiple images
        assert args.skip_multi_images, "Only support single image question for now"
        if args.skip_multi_images:
            dataset = dataset.filter(lambda x: x['image_2'] is None)

        # print("check")
        # for entry in tqdm(dataset):
        #     image_order = [int(num) for num in re.findall(r'<image\s+(\d+)>', entry['question'])]
        #     if len(image_order) > 1:
        #         breakpoint()

        progress = tqdm(dataset)

        results[test_type] = []
        for entry in progress:
            prompt, images = process_prompt(entry, test_type)
            # if test_type == "vision":
            # TODO: single image for now
            images = images[:1]
            for _ in images:
                prompt = "<image>\n" + prompt
            # cnt_of_image_in_prompt = prompt.count("<image>")
            # try:
            #     assert cnt_of_image_in_prompt == 1
            # except:
            #     breakpoint()
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()


            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = process_images(images, image_processor, model.config)

            output = model.generate(input_ids, images=image_tensor.half().cuda(), max_new_tokens=1024,
                                    return_dict_in_generate=True, do_sample=False, temperature=None, top_p=None)
            decoded_output = tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)[0].strip()

            # Analyze attention
            outputs_for_attn_analysis = model(input_ids, use_cache=True, images=image_tensor.half().cuda(), output_attentions=True)

            batch_size = input_ids.size(0)
            assert batch_size == 1
            last_token_attn_scores = outputs_for_attn_analysis.attentions[-1][0, :, -1, :]
            avg_last_token_attn_scores = torch.mean(last_token_attn_scores, dim=0)
            all_image_token_indices = outputs_for_attn_analysis.all_image_token_indices[0]
            last_token_to_all_image_token_attn_scores = avg_last_token_attn_scores[all_image_token_indices].cpu().tolist()
            CLS_tok_image_attentions = outputs_for_attn_analysis.image_attentions[0].cpu().tolist()

            correct = decoded_output == entry['answer']

            # if not correct:
            #     breakpoint()

            per_type_acc[test_type].append(correct)
            all_acc.append(correct)

            result_single_sample = {}
            result_single_sample['id'] = entry["id"]
            result_single_sample['question'] = entry['question'] if test_type != "vision" else ""
            result_single_sample['options'] = literal_eval(entry['options'])
            result_single_sample['option_chosen'] = decoded_output
            result_single_sample['correct'] = correct
            result_single_sample['last_token_to_all_image_token_attn_scores'] = last_token_to_all_image_token_attn_scores
            result_single_sample['CLS_tok_image_attentions'] = CLS_tok_image_attentions
            test_type_last_token_to_all_image_token_attn_scores.append(last_token_to_all_image_token_attn_scores)
            try:
                if len(test_type_last_token_to_all_image_token_attn_scores) > 1:
                    assert len(last_token_to_all_image_token_attn_scores) == len(test_type_last_token_to_all_image_token_attn_scores[-2])
            except:
                breakpoint()
            test_type_CLS_tok_image_attentions.append(CLS_tok_image_attentions)
            all_last_token_to_all_image_token_attn_scores.append(last_token_to_all_image_token_attn_scores)
            all_CLS_tok_image_attentions.append(CLS_tok_image_attentions)

            results[test_type].append(result_single_sample)

            progress.set_description(f"{test_type} Acc: {np.mean(per_type_acc[test_type]) * 100:.1f}%")

        with open(os.path.join(args.output_dir, f"answers_{test_type}{'_skip_multi_image' if args.skip_multi_images else ''}.json"), 'w') as f:
            json.dump(results[test_type], f, indent=4)
        print(test_type, np.mean(per_type_acc[test_type]))
        acc_str = f"{test_type} Accuracy: {np.mean(per_type_acc[test_type]) * 100:.1f}% ({sum(per_type_acc[test_type])}/{len(per_type_acc[test_type])})"
        with open(os.path.join(args.output_dir, f"accuracy{'_skip_multi_image' if args.skip_multi_images else ''}.txt"), "a") as f:
            f.write(acc_str + "\n")

        # Visualize avg all_last_token_to_all_image_token_attn_scores and all_CLS_tok_image_attentions
        try:
            avg_last_token_to_all_image_token_attn_scores = np.average(
                np.array(test_type_last_token_to_all_image_token_attn_scores), axis=0)
        except:
            breakpoint()
        visualize_token_to_vis_token_attn_scores(avg_last_token_to_all_image_token_attn_scores, "Last Text To Image Token Attn Score", os.path.join(args.output_dir, f"{test_type}_last_txt_to_image_attn_score.png"))
        avg_CLS_tok_image_attentions = np.average(np.array(test_type_CLS_tok_image_attentions), axis=0)
        visualize_token_to_vis_token_attn_scores(avg_CLS_tok_image_attentions, "CLS To Image Token Attn Score", os.path.join(args.output_dir, f"{test_type}_CLS_image_attn_score.png"))

    # Visualize avg all_last_token_to_all_image_token_attn_scores and all_CLS_tok_image_attentions
    avg_last_token_to_all_image_token_attn_scores = np.average(np.array(all_last_token_to_all_image_token_attn_scores), axis=0)
    visualize_token_to_vis_token_attn_scores(avg_last_token_to_all_image_token_attn_scores,"Last Text To Image Token Attn Score", os.path.join(args.output_dir, f"overall_last_txt_to_image_attn_score.png"))
    avg_CLS_tok_image_attentions = np.average(np.array(all_CLS_tok_image_attentions), axis=0)
    visualize_token_to_vis_token_attn_scores(avg_CLS_tok_image_attentions, "CLS To Image Token Attn Score", os.path.join(args.output_dir, f"overall_CLS_image_attn_score.png"))

    print(np.mean(all_acc))
    acc_str = f"Overall Accuracy: {np.mean(all_acc) * 100:.1f}% ({sum(all_acc)}/{len(all_acc)})"
    with open(os.path.join(args.output_dir, f"accuracy{'_skip_multi_image' if args.skip_multi_images else ''}.txt"), "a") as f:
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
    parser.add_argument("--skip_multi_images", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_model(args)
