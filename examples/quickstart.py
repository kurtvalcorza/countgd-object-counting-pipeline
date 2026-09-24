"""Count the tutorial's synthetic demo scene three ways: by text, by three exemplar boxes, and by both.

Run from a checkout after `python scripts/fetch_weights.py` (or let `from_pretrained` fetch and convert the
pinned checkpoint into the cache on first use)."""
from countgd_pipeline import DEMO_SEED, CountGDPipeline, synthetic_scene


def main() -> None:
    pipe = CountGDPipeline.from_pretrained()
    scene = synthetic_scene(DEMO_SEED)
    print(f"gold: {scene['count']} x {scene['label']!r} (and {scene['distractors']['count']} distractors)")
    for name, kwargs in (
        ("text", {"text": scene["label"]}),
        ("exemplars", {"exemplars": scene["exemplars"]}),
        ("text + exemplars", {"text": scene["label"], "exemplars": scene["exemplars"]}),
    ):
        entry = pipe.count(scene["image"], **kwargs)["results"][0]
        print(f"{name:>16}: {entry['count']} (max score {entry['max_score']:.2f})")


if __name__ == "__main__":
    main()
