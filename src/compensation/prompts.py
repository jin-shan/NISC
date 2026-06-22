from __future__ import annotations

import re


def scenario_prompt(word: str, scenario_count: int) -> str:
    return (
        "You are a professional natural language processing data augmentation expert.\n"
        "This study focuses on bias mitigation for a Chinese toxic language detection model.\n"
        "We find that the model may form a spurious association with a specific term and may misclassify neutral texts containing this term as toxic.\n"
        f"Target term: {word}\n"
        f"Please design {scenario_count} different Chinese social-media text generation scenarios around the target term.\n"
        "The scenarios should be diverse and should cover contexts such as daily discussions, news comments, life descriptions, objective introductions, personal experiences, public services, and educational explanations.\n"
        "Requirements:\n"
        "1. Each scenario must be suitable for generating Chinese texts containing the target term.\n"
        "2. The scenarios must not contain attack, insult, discrimination, hatred, or incitement.\n"
        "3. The scenarios should be close to real expressions in Chinese social media environments such as Weibo, Zhihu, and Tieba.\n"
        "4. Please directly output the scenario instructions in a list format."
    )


def sample_prompt(word: str, scenario: str, sample_count: int) -> str:
    return (
        "You are a data generation assistant familiar with Chinese internet language.\n"
        f"Scenario instruction: {scenario}\n"
        f"Target term: {word}\n"
        f"Please generate {sample_count} Chinese short texts.\n"
        "Requirements:\n"
        "1. Each text must naturally contain the target term.\n"
        "2. The texts should be authentic and natural.\n"
        "3. Each text should contain 20 to 60 Chinese characters.\n"
        "4. The texts must be safe, harmless, and non-toxic.\n"
        "5. Do not include insult, attack, discrimination, hatred, threat, incitement, or derogatory content.\n"
        "6. Please directly output the texts in a list format."
    )


def parse_list_output(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^\s*[\-\*\d\.\)\(]+\s*", "", line).strip()
        if cleaned:
            items.append(cleaned)
    return items
