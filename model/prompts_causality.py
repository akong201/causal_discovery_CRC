# model/prompts_causality.py

system_prompt = "You are an expert in scientific reasoning and causality."

fewshot_inst_prompt = """You will be given a 'Claim' and a 'Document'. Your task is to determine if the document provides evidence to prove or disprove the causal relationship asserted in the claim.

- If the document **proves** the causal relationship, respond with [[True]].
- If the document **disproves** or **falsifies** the causal relationship, respond with [[False]].
- Do not provide any explanation, only the verdict.

Here are some examples:"""

fewshot_example_prompt = """
[Claim]
{claim}

[Document]
{document}

[Verdict]
{verdict}
"""

fewshot_query_prompt = """
[Claim]
{claim}

[Document]
{document}

[Verdict]"""
