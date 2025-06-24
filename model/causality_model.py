# model/causality_model.py

import math
import os
import time
from typing import List, Dict, Tuple

import openai
from openai.types.chat.chat_completion import Choice
from wrapt_timeout_decorator import timeout

from .prompts_causality import system_prompt, fewshot_inst_prompt, fewshot_example_prompt, fewshot_query_prompt

def set_openai_api_key():
    """
    Checks for and sets the OpenAI API key from environment variables.
    Raises an error if the key is not found.
    """
    if not (api_key := os.getenv("OPENAI_API_KEY")):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key
    return api_key

class CausalityModel:
    """
    A wrapper for the OpenAI API to evaluate causal claims based on a document.
    It uses few-shot prompting to guide the model and extracts log probabilities
    to determine the model's confidence.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.Client(api_key=set_openai_api_key())
        # The tokens we expect the model to evaluate
        self.target_tokens = ["True", "False"]

    def evaluate_causality(self, sample: Dict, fewshot_examples: List[Dict]) -> Tuple[str | None, Dict | None]:
        """
        Evaluates a single sample using few-shot prompting.

        Args:
            sample (Dict): A dictionary containing the 'claim' and 'document'.
            fewshot_examples (List[Dict]): A list of examples for the few-shot prompt.

        Returns:
            A tuple containing:
            - The predicted result ("True" or "False").
            - A dictionary with the probabilities for "True" and "False".
        """
        prompt_config = {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": 5,
            "logprobs": True,
            "top_logprobs": 5, # Request top 5 logprobs for each token position
        }

        system, user_prompt = self.prepare_fewshot_prompt(sample['claim'], sample['document'], fewshot_examples)
        
        output = self.prompt_generation(system, user_prompt, prompt_config)

        if output is None:  # Handle API errors
            return None, None

        # --- Prediction and Probability Extraction ---
        prediction = self.extract_prediction(output.message.content)
        probabilities = self.extract_probabilities(output)
        
        if not prediction or not probabilities:
            return None, None
            
        return prediction, probabilities

    @staticmethod
    def prepare_fewshot_prompt(claim: str, document: str, fewshot_examples: List[Dict]) -> Tuple[str, str]:
        """Builds the full user-facing prompt from templates and examples."""
        prompt = fewshot_inst_prompt

        for example in fewshot_examples:
            verdict = "[[True]]" if example["is_causal"] else "[[False]]"
            prompt += "\n" + fewshot_example_prompt.format(
                claim=example["claim"],
                document=example["document"],
                verdict=verdict,
            )

        prompt += "\n" + fewshot_query_prompt.format(claim=claim, document=document)
        return system_prompt, prompt

    def extract_prediction(self, generated_text: str) -> str | None:
        """Extracts [[True]] or [[False]] from the model's text output."""
        text = generated_text.strip()
        if "[[True]]" in text:
            return "True"
        if "[[False]]" in text:
            return "False"
        return None

    def extract_probabilities(self, output: Choice) -> Dict | None:
        """
        Extracts the log probabilities for the target tokens ("True", "False")
        from the complex logprobs object returned by the API.
        """
        result_log_probs = {}
        
        # Search for the first token after the opening brackets '[['
        try:
            token_list = [x.token for x in output.logprobs.content]
            # Find the index of the token that starts the verdict
            start_index = -1
            for i, token in enumerate(token_list):
                if '[[' in token and i + 1 < len(token_list):
                    start_index = i + 1
                    break
            
            if start_index == -1:
                return None

            # Get the logprobs for the token position right after '[['
            top_logprobs = output.logprobs.content[start_index].top_logprobs
            
            for top_logprob in top_logprobs:
                if top_logprob.token in self.target_tokens:
                    result_log_probs[top_logprob.token] = top_logprob.logprob

            # If a target token wasn't in the top_logprobs, assign it a very low probability
            for token in self.target_tokens:
                if token not in result_log_probs:
                    result_log_probs[token] = -100 # A very small log probability

            # Normalize to get probabilities
            total = sum(math.exp(lp) for lp in result_log_probs.values())
            probabilities = {k: math.exp(v) / total for k, v in result_log_probs.items()}
            
            return probabilities

        except (AttributeError, IndexError):
            return None

    @timeout(20, timeout_exception=StopIteration)
    def _request_generation(self, system_prompt: str, prompt: str, prompt_config: Dict) -> Choice:
        """Internal method to make the API call with a timeout."""
        completion = self.client.chat.completions.create(
            **prompt_config,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0]

    def prompt_generation(self, system_prompt: str, prompt: str, prompt_config: Dict) -> Choice | None:
        """

        Makes the API call and handles retries and common errors.
        """
        try:
            return self._request_generation(system_prompt, prompt, prompt_config)
        except openai.BadRequestError as e:
            print(f"Caught a BadRequestError: {e}")
            return None
        except (openai.OpenAIError, OSError, StopIteration) as e:
            retry_time = 1
            if hasattr(e, 'retry_after'):
                retry_time = e.retry_after
            print(f"Server error or timeout. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            try:
                # One last retry attempt
                return self._request_generation(system_prompt, prompt, prompt_config)
            except Exception as final_e:
                print(f"Final retry attempt failed: {final_e}")
                return None

