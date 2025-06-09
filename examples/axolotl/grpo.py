from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

SYSTEM_PROMPT = SYSTEM_PROMPTS["DeepSeekZero"] # or define your own system prompt


def axo_rg_transform(cfg, *args, **kwargs):
    # first create and push a dataset (see scripts/hf_dataset/save_hf_dataset.py)

    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + example["question"]},
            ],
            "answer": example["answer"],
            "metadata": example["metadata"],
        }

    return transform_fn, {}


# Reward functions
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    print(kwargs)
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
