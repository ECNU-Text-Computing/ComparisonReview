import asyncio
from openai import AsyncOpenAI
from config import API_URL, API_KEY
from dataclasses import dataclass
from typing import Optional, Tuple
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from paper import Paper
import json


@dataclass(slots=True)
class TokenCounter:
    input_tokens: int = 0,
    output_tokens: int = 0


@dataclass
class PredictorConfig:
    model_path: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    do_sample: bool = False


class Predictor:
    def __init__(self, config: PredictorConfig = None, **kwargs):
        self.config = config
        self.device = 'cuda'
        self.model_path = config.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="cuda:0",
            trust_remote_code=True
        )
        self.model.eval()

    def predict(self, prompts: list[str]) -> Tuple[list[str], list[TokenCounter]]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        completions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
        output_lengths = [(generated_tokens[i] != self.tokenizer.pad_token_id).sum().item()
                          for i in range(generated_tokens.size(0))]
        token_counts = [
            TokenCounter(input_tokens=input_length, output_tokens=output_length)
            for input_length, output_length in zip(input_lengths, output_lengths)
        ]
        return completions, token_counts


@dataclass
class OpenaiPredictorConfig:
    api_base: str
    model_name: str
    api_key: Optional[str] = None
    dataset: Optional[str] = None
    max_new_tokens: int = 1024
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: bool = False


class OpenaiPredictor:
    def __init__(self, config: OpenaiPredictorConfig):
        self.config = config
        api_key = config.api_key
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=api_key,
        )

    async def _predict(self, prompts: list[str]) -> Tuple[list[str], list[TokenCounter]]:
        async def _single_request(prompt: str) -> Tuple[str, TokenCounter]:
            resp = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=None
            )
            token_count = TokenCounter(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens
            )
            return resp.choices[0].message.content, token_count
        tasks = [_single_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        responses, token_counts = zip(*results)
        return list(responses), list(token_counts)

    def predict(self, prompts: list[str]) -> tuple[list[str], list[TokenCounter]]:
        return asyncio.run(self._predict(prompts))


class LLMJudger:
    def __init__(
            self,
            model: str,
            online: bool = True,
    ):
        if not online:
            self.pred_config = PredictorConfig(
                model_path=model,
                max_new_tokens=512,
                temperature=None,
                top_p=None,
                top_k=None,
                do_sample=False
            )
            self.predictor = Predictor(self.pred_config)
        else:
            self.pred_config= OpenaiPredictorConfig(
                api_base=API_URL,
                model_name=model,
                api_key=API_KEY,
                temperature=None,
                top_p=None,
                top_k=None,
                do_sample=False
            )
            self.predictor = OpenaiPredictor(self.pred_config)


    @staticmethod
    def create_prompt(paper1: Paper, paper2: Paper) -> str:
        prefix = "Your response must be about 200 words in length. "
        prompt = """
Please act as an impartial judge and evaluate the quality of the following two papers. As the area chair for a top ML conference, you can only select one paper. Start with a brief meta-review / reasoning of the pros and cons for each paper (two sentences), and then provide your choice in a binary format. Start with a brief meta-review / reasoning of the pros and cons for each paper, focusing on novelty, significance, clarity, methodology, and practical implications. Be very critical and do not be biased by what the author claimed. Finally, provide your choice in a binary format.

Please provide your analysis in JSON format.

### Paper 1:
Submission Title: {title_1}
```
Abstract: {abstract_1}
```

### Paper 2:
Submission Title: {title_2}
```
Abstract: {abstract_2}
```

Your JSON output should look like this:
{{
  "paper_1_review": "Your meta-review and reasoning for paper 1",
  "paper_2_review": "Your meta-review and reasoning for paper 2",
  "chosen_paper": "paper_1 or paper_2"
}}
"""
        input_data = {
            "title_1": paper1.title,
            "abstract_1": paper1.abstract,
            "title_2": paper2.title,
            "abstract_2": paper2.abstract
        }
        return prefix + prompt.format(**input_data)

    @staticmethod
    def extract_answer(raw_answer: str) -> int:
        try:
            result = json.loads(raw_answer)
            chosen = result.get("chosen_paper", "").strip().lower()
            if chosen == "paper_1":
                return 1
            elif chosen == "paper_2":
                return 0
            else:
                return -1
        except Exception:
            match = re.search(r'"chosen_paper":\s*"paper_([12])"', raw_answer)
            if match:
                if match.group(1) == "1":
                    return 1
                elif match.group(1) == "2":
                    return 0
            return -1

    def batch_compare(self, papers: list[Tuple[Paper, Paper]]) -> Tuple[list[int], list[str], list[TokenCounter]]:
        prompts = []
        for paper1, paper2 in papers:
            prompt = self.create_prompt(paper1, paper2)
            prompts.append(prompt)

        with torch.no_grad():
            answers, token_counts = self.predictor.predict(prompts)
        preferences = []
        for answer in answers:
            answer = str(answer)
            choice = self.extract_answer(answer)
            preferences.append(choice)
        return preferences, answers, token_counts
