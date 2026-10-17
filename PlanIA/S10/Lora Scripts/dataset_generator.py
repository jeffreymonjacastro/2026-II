"""
Training & Evaluation Dataset Generator
========================================
File created by Javier Ideami (ideami.com)

Generates synthetic customer-support examples for fine-tuning.
Each example is an (instruction, input, output-JSON) triple that
covers all intents, urgencies, product/null variants, and edge cases.
"""

import json
import random
from typing import Dict, List, Tuple

# ── Shared instruction (used for every example) ────────────────────────────

SYSTEM_INSTRUCTION = (
    "You are a customer support assistant. Extract structured information from "
    "the customer message and respond with ONLY a JSON object (no additional "
    "text) following this exact schema:\n"
    '{"intent": "refund|replacement|technical_support|shipping|other", '
    '"product": "string or null", "urgency": "low|medium|high", '
    '"needs_human": true or false, "summary": "brief summary in 5-25 words"}'
)

# ── Hand-written seed examples (one per key scenario) ──────────────────────

_SEED_EXAMPLES: List[Dict] = [
    # --- Refund ---
    {"input": "I want my money back for the broken laptop I received. This is urgent!",
     "output": {"intent": "refund", "product": "laptop", "urgency": "high",
                "needs_human": True, "summary": "Customer requests urgent refund for broken laptop received"}},
    {"input": "Hi, I'd like to return the headphones I bought last week and get a refund please.",
     "output": {"intent": "refund", "product": "headphones", "urgency": "low",
                "needs_human": False, "summary": "Customer wants to return headphones and receive refund"}},
    {"input": "The blue sweater doesn't fit. Can I get my money back?",
     "output": {"intent": "refund", "product": "blue sweater", "urgency": "medium",
                "needs_human": False, "summary": "Customer requests refund for sweater that does not fit"}},
    # --- Replacement ---
    {"input": "My phone arrived damaged. I need a replacement as soon as possible!",
     "output": {"intent": "replacement", "product": "phone", "urgency": "high",
                "needs_human": True, "summary": "Customer needs urgent replacement for damaged phone received"}},
    {"input": "Can I exchange this book for a different edition?",
     "output": {"intent": "replacement", "product": "book", "urgency": "low",
                "needs_human": False, "summary": "Customer wants to exchange book for different edition"}},
    # --- Technical support ---
    {"input": "HELP! My smart watch won't turn on and I've tried everything!",
     "output": {"intent": "technical_support", "product": "smart watch", "urgency": "high",
                "needs_human": True, "summary": "Customer needs urgent technical help with smart watch not turning on"}},
    {"input": "How do I connect my wireless mouse to my computer?",
     "output": {"intent": "technical_support", "product": "wireless mouse", "urgency": "low",
                "needs_human": False, "summary": "Customer needs help connecting wireless mouse to computer"}},
    {"input": "The app keeps crashing when I try to log in. Can someone help?",
     "output": {"intent": "technical_support", "product": "app", "urgency": "medium",
                "needs_human": False, "summary": "Customer experiencing app crashes during login and needs assistance"}},
    # --- Shipping ---
    {"input": "Where is my package? It was supposed to arrive 3 days ago!",
     "output": {"intent": "shipping", "product": None, "urgency": "high",
                "needs_human": False, "summary": "Customer inquiring about delayed package not received as scheduled"}},
    {"input": "Just checking on the status of my order #12345",
     "output": {"intent": "shipping", "product": None, "urgency": "low",
                "needs_human": False, "summary": "Customer checking status of their order number twelve thousand"}},
    {"input": "I need to change my delivery address before the tablet ships.",
     "output": {"intent": "shipping", "product": "tablet", "urgency": "medium",
                "needs_human": True, "summary": "Customer needs to update delivery address before tablet shipment"}},
    # --- Other / ambiguous ---
    {"input": "I have a question about your return policy.",
     "output": {"intent": "other", "product": None, "urgency": "low",
                "needs_human": False, "summary": "Customer inquiring about the company return policy details"}},
    {"input": "What are your business hours?",
     "output": {"intent": "other", "product": None, "urgency": "low",
                "needs_human": False, "summary": "Customer asking about business operating hours"}},
    {"input": "I'm very upset with the service I received from your support team!",
     "output": {"intent": "other", "product": None, "urgency": "high",
                "needs_human": True, "summary": "Customer expressing strong dissatisfaction with support team service"}},
]

# ── Templates and product lists for programmatic variation ─────────────────

_PRODUCTS = {
    "refund": ["camera", "keyboard", "monitor", "tablet", "shoes", "jacket", "coffee maker", "backpack"],
    "replacement": ["charger", "cable", "speaker", "webcam", "chair", "desk lamp", "water bottle"],
    "technical_support": ["router", "printer", "scanner", "TV", "gaming console", "fitness tracker"],
    "shipping": [None, "package", "order", "delivery"],
}

_TEMPLATES = {
    "refund": [
        "I need a refund for the {product}. It's not what I expected.",
        "Please refund my purchase of the {product}. Not satisfied.",
        "The {product} is defective. I want my money back immediately!",
        "Can I get a refund? The {product} stopped working after one day.",
    ],
    "replacement": [
        "I'd like to replace the {product} with a new one.",
        "The {product} has a defect. Can you send a replacement?",
        "Need a replacement for my {product} ASAP!",
        "Is it possible to get a new {product}? This one is broken.",
    ],
    "technical_support": [
        "My {product} isn't working properly. Help!",
        "I can't figure out how to use the {product}.",
        "The {product} has an error message. What should I do?",
        "Need technical assistance with my {product}.",
    ],
    "shipping": [
        "When will my {product} arrive?",
        "Track my {product} order please.",
        "My {product} shipment is late!",
        "Update on {product} delivery?",
    ],
}

_URGENCIES = ["low", "medium", "high"]


def _generate_variations() -> List[Dict]:
    """Create many examples programmatically from templates."""
    variations: List[Dict] = []

    # Refund / replacement / tech-support: 2 templates × each product
    for intent in ("refund", "replacement", "technical_support"):
        human_base_rate = 0.4 if intent == "technical_support" else 0.3
        for product in _PRODUCTS[intent]:
            for template in _TEMPLATES[intent][:2]:
                urgency = random.choice(_URGENCIES)
                # For refunds and replacements, we want a mix of human-needed and not, with higher urgency more likely to need human. For tech support, we want an even higher base rate of human-needed.
                needs_human = urgency == "high" or random.random() < human_base_rate
                variations.append({
                    "input": template.format(product=product),
                    "output": {
                        "intent": intent,
                        "product": product,
                        "urgency": urgency,
                        "needs_human": needs_human,
                        # The summary should be a natural language description of the customer's request, incorporating the intent, urgency, and product in a concise way. The replace is to make the intent more human-readable in the summary (e.g. "technical_support" becomes "technical support").
                        "summary": (f"Customer requests {urgency} urgency {intent.replace('_', ' ')} "
                                    f"for {product}"),
                    },
                })

    # Shipping: 20 random combos
    for template in _TEMPLATES["shipping"] * 5:
        product = random.choice(_PRODUCTS["shipping"])
        urgency = random.choice(_URGENCIES)
        # For shipping, we want a mix of human-needed and not, but only for high urgency cases, and even then only about half the time, since many shipping inquiries can be handled by the system unless they are very urgent.
        needs_human = urgency == "high" and random.random() < 0.5

        if product is None:
            # For the null product case, we want to test that the model can handle when the product is not explicitly mentioned. We can modify the template to remove the product mention, and adjust the summary accordingly.
            input_text = template.replace("my {product}", "my order").replace("{product}", "")
            summary = f"Customer inquiring about shipment with {urgency} urgency"
        else:
            input_text = template.format(product=product)
            summary = f"Customer asking about {product} shipping status {urgency} priority"

        variations.append({
            "input": input_text,
            "output": {"intent": "shipping", "product": product,
                       "urgency": urgency, "needs_human": needs_human,
                       "summary": summary},
        })

    return variations


# ── Public API ─────────────────────────────────────────────────────────────

def generate_training_dataset(
    train_size: int = 400,
    eval_size: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build train + eval splits of instruction-following examples.
    Tuple[List[Dict], List[Dict]] - A tuple containing exactly two elements, 
    where both elements are lists of dictionaries (Training and Evaluation datasets).
    Each returned dict has keys: instruction, input, output (JSON string).
    """

    # Mix seed examples with programmatically generated variations, 
    # then shuffle and split
    all_examples = _SEED_EXAMPLES + _generate_variations()
    random.shuffle(all_examples)

    total_needed = train_size + eval_size
    if len(all_examples) < total_needed:
        multiplier = (total_needed // len(all_examples)) + 1
        # If we don't have enough unique examples, repeat the list until 
        # we do, then shuffle again
        all_examples = (all_examples * multiplier)[:total_needed]
        random.shuffle(all_examples)
    else:
        all_examples = all_examples[:total_needed]

    formatted = [
        {
            "instruction": SYSTEM_INSTRUCTION,
            "input": ex["input"],
            "output": json.dumps(ex["output"], ensure_ascii=False),
        }
        for ex in all_examples
    ]

    return formatted[:train_size], formatted[train_size:]
