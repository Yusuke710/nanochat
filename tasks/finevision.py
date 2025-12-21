"""
FineVision - Wrapper for HuggingFace FineVision dataset subsets.

FineVision unifies 200+ sources into 185 subsets with 24.3M samples.
Use this to load any subset and integrate with TaskMixture.

Usage:
    from tasks.finevision import FineVision
    from tasks.common import TaskMixture

    train_ds = TaskMixture([
        FineVision("DoclingMatix", stop=500_000),
        FineVision("SynthChartNet", stop=300_000),
        FineVision("latexformulas", stop=100_000),
    ])

FineVision format (input):
    {
        "images": [PIL.Image],
        "texts": [{"user": "...", "assistant": "..."}],
        "source": "DoclingMatix",
    }

Nanochat format (output):
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "images": [PIL.Image],  # PIL images directly
    }
"""

from datasets import load_dataset
from tasks.common import Task


class FineVision(Task):
    """Wrapper for any FineVision subset.

    Uses HuggingFace datasets with Arrow memory-mapping for efficient random access.
    Data is cached on disk, not in RAM.

    Args:
        subset_name: Name of the FineVision subset (e.g., "DoclingMatix", "SynthChartNet")
        **kwargs: Passed to Task base class (start, stop, step for slicing)

    Examples:
        FineVision("chartqa")                    # full dataset
        FineVision("chartqa", stop=10000)        # first 10K samples
        FineVision("chartqa", start=10000, stop=10100)  # samples 10000-10099
    """

    def __init__(self, subset_name: str, **kwargs):
        super().__init__(**kwargs)
        self.subset_name = subset_name

        # Load subset with HF native random access (memory-mapped Arrow files)
        print(f"Loading FineVision/{subset_name}...")
        self.ds = load_dataset(
            "HuggingFaceM4/FineVision",
            name=subset_name,
            split="train",
        )

        print(f"  Ready: {len(self.ds)} samples from {subset_name}")

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        sample = self.ds[index]
        texts = sample["texts"]

        # Handle multi-turn conversations (flatten to first turn for now)
        # TODO: Support multi-turn if needed
        if isinstance(texts, list) and len(texts) > 0:
            first_turn = texts[0]
            user_content = first_turn.get("user", "")
            assistant_content = first_turn.get("assistant", "")
        else:
            user_content = ""
            assistant_content = ""

        # Add <image> placeholder if images present
        images = sample.get("images", [])
        if images:
            user_content = f"<image>\n{user_content}"

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": images,  # PIL images directly
        }


if __name__ == "__main__":
    # Quick test
    from datasets import get_dataset_config_names

    # List available subsets
    subsets = get_dataset_config_names("HuggingFaceM4/FineVision")
    print(f"Available FineVision subsets: {len(subsets)}")
    print(f"First 10: {subsets[:10]}")

    # Test loading a small subset
    print("\nTesting chartqa subset...")
    ds = FineVision("chartqa", stop=5)
    print(f"Loaded {len(ds)} samples")

    example = ds[0]
    print(f"Example keys: {example.keys()}")
    print(f"User: {example['messages'][0]['content'][:100]}...")
    print(f"Assistant: {example['messages'][1]['content'][:100]}...")
    print(f"Has images: {len(example.get('images', []))}")
