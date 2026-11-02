"""Datasets as `Sample` records: manifests, MedEvalKit prompts and bundle builders (plan.md §C2, Stage 2).

CLI (identical on Kaggle and SSH):
    python -m medprune.data build   --out DIR [--work DIR] [--skip radimagenet]
    python -m medprune.data masks   --bundle DIR
    python -m medprune.data summary --bundle DIR
RadImageNet-VQA may not be redistributed: build it inside each GPU session with `--only radimagenet`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    qid: str  # globally unique, e.g. "slake_en/1234"
    dataset: str  # dataset id (plan.md §C2)
    image_id: str  # bootstrap cluster key
    image_path: str  # relative to the bundle root
    question: str
    answer: str  # multiple choice: the letter
    answer_type: str  # "open" | "closed" | "yesno" | "mc" — decides the scorer
    prompt_style: str  # "close" | "open" | "mc" | "judgement" — decides the MedEvalKit prompt
    choices: tuple[str, ...] | None = None  # multiple choice only, formatted "A. text"
    lang: str = "en"
    modality: str = ""
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- prompts

def make_prompt(sample: Sample) -> str:
    """MedEvalKit utils/question_formats.py, non-reasoning variants (English and Chinese)."""
    q, zh = sample.question, sample.lang == "zh"
    if sample.prompt_style == "mc":
        options = "\n".join(sample.choices or ())
        if zh:
            return f"\n问题： {q}\n选项： \n{options}" + "\n" + "请直接使用给定选项中的选项字母来回答该问题。"
        return f"\nQuestion: {q}\nOptions: \n{options}" + "\n" + "Answer with the option's letter from the given choices directly."
    if sample.prompt_style == "judgement":
        return q + "\n" + ("请输出'是'或'否'(不要有任何其它输出)。" if zh else "Please output 'yes' or 'no'(no extra output).")
    if sample.prompt_style == "close":
        return q + "\n" + ("请用一个单词或者短语回答该问题。" if zh else "Answer the question using a single word or phrase.")
    if sample.prompt_style == "open":
        return q + "\n" + ("请简要回答该问题。" if zh else "Please answer the question concisely.")
    raise ValueError(f"unknown prompt_style {sample.prompt_style!r}")


# --------------------------------------------------------------------------------------------- manifests

def write_manifest(samples: list[Sample], root: Path, dataset: str) -> Path:
    path = Path(root) / "manifests" / f"{dataset}.jsonl"
    validate_question_ids(samples, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return path


def load_manifest(root: Path, dataset: str) -> list[Sample]:
    samples = []
    with (Path(root) / "manifests" / f"{dataset}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["choices"] = tuple(row["choices"]) if row["choices"] is not None else None
            samples.append(Sample(**row))
    validate_question_ids(samples, f"{root}/manifests/{dataset}.jsonl")
    return samples


def validate_question_ids(samples: list[Sample], source: str) -> None:
    """Refuse template IDs reused across examples before resume or aggregation can lose records."""
    seen = set()
    for sample in samples:
        if sample.qid in seen:
            raise ValueError(f"Duplicate qid {sample.qid!r} in {source}; question IDs must identify individual examples")
        seen.add(sample.qid)


def _letter_choices(options: list[str], answer_text: str) -> tuple[tuple[str, ...], str | None]:
    """Format options as "A. text" and return the letter whose text equals the answer (None if no match)."""
    choices, letter = [], None
    for i, text in enumerate(options):
        tag = chr(ord("A") + i)
        choices.append(f"{tag}. {text}")
        if text == answer_text:
            letter = tag
    return tuple(choices), letter


# ------------------------------------------------------------------------------------------------- SLAKE

def build_slake(raw_dir: Path, out_root: Path) -> dict[str, int]:
    """SLAKE test split → slake_en, slake_zh. `raw_dir` holds test.json and the unzipped imgs/ folder."""
    rows = json.loads((Path(raw_dir) / "test.json").read_text(encoding="utf-8"))
    by_lang: dict[str, list[Sample]] = defaultdict(list)
    folders = set()
    for r in rows:
        lang, answer, folder = r["q_lang"], str(r["answer"]), r["img_name"].split("/")[0]
        if r["answer_type"] == "OPEN":
            answer_type, style = "open", "open"
        else:  # MedEvalKit: yes/no answers use the yes/no judge, other closed answers use exact match
            answer_type, style = ("yesno" if answer in ("yes", "no") else "closed"), "close"
        by_lang[lang].append(Sample(
            qid=f"slake_{lang}/{r['qid']}", dataset=f"slake_{lang}", image_id=f"slake/{folder}",
            image_path=f"images/slake/{r['img_name']}", question=r["question"], answer=answer,
            answer_type=answer_type, prompt_style=style, lang=lang, modality=r["modality"],
            meta={"content_type": r["content_type"], "location": r["location"], "img_id": r["img_id"]},
        ))
        folders.add(folder)
    for folder in sorted(folders):  # images with mask.png and detection.json (oracle gate, evidence sizes)
        shutil.copytree(Path(raw_dir) / "imgs" / folder, Path(out_root) / "images" / "slake" / folder, dirs_exist_ok=True)
    counts = {}
    for lang, samples in by_lang.items():
        write_manifest(samples, out_root, f"slake_{lang}")
        counts[f"slake_{lang}"] = len(samples)
    return counts


# ------------------------------------------------------------------------------------------- OmniMedVQA

# Keywords that map OmniMedVQA's raw modality_type strings to our panel names (raw strings are printed at build).
PANEL_MODALITIES = {
    "CT": ("computed tomography",),
    "MRI": ("magnetic", "mag-netic"),
    "X-Ray": ("x-ray",),
    "Ultrasound": ("ultrasound",),
    "OCT": ("optical coherence",),
    "Fundus": ("fundus",),
    "Dermoscopy": ("dermoscopy",),
    "Microscopy": ("microscopy",),
}


def normalize_modality(raw: str) -> str:
    low = raw.lower()
    for name, keys in PANEL_MODALITIES.items():
        if any(k in low for k in keys):
            return name
    return raw


def build_omnimedvqa(zip_path: Path, out_root: Path, panel_per_modality: int = 150, max_per_image: int = 3,
                     seed: int = 0, anat_size: int = 3329) -> dict[str, int]:
    """OmniMedVQA open-access → omni_anat (Anatomy Identification) and omni_panel (8 modalities x 150)."""
    with zipfile.ZipFile(zip_path) as zf:
        qa_files = [n for n in zf.namelist() if "QA_information/Open-access/" in n and n.endswith(".json")]
        if not qa_files:
            raise FileNotFoundError("no QA_information/Open-access/*.json inside the zip")
        prefix = qa_files[0].split("QA_information/")[0]
        items = [item for n in qa_files for item in json.loads(zf.read(n))]
        for it in items:  # one source file names the field "modality" on some rows
            it.setdefault("modality_type", it.get("modality", "unknown"))
        raw_modalities = Counter(it["modality_type"] for it in items)
        print("OmniMedVQA open-access items:", len(items))
        print("modality_type values:", dict(raw_modalities))

        def to_sample(it: dict, dataset: str) -> Sample | None:
            options = [it[f"option_{c}"] for c in "ABCD" if f"option_{c}" in it]
            choices, letter = _letter_choices(options, it["gt_answer"])
            if letter is None:
                return None
            return Sample(
                qid=f"omni/{it['question_id']}", dataset=dataset, image_id=f"omni/{it['image_path']}",
                image_path=f"images/omnimedvqa/{it['image_path']}", question=it["question"], answer=letter,
                answer_type="mc", prompt_style="mc", choices=choices, modality=normalize_modality(it["modality_type"]),
                meta={"question_type": it["question_type"], "source": it["dataset"], "modality_raw": it["modality_type"]},
            )

        anat = [s for it in items if it["question_type"] == "Anatomy Identification"
                if (s := to_sample(it, "omni_anat")) is not None]
        anat_ids = {s.qid for s in anat}  # the panel stays disjoint from every anatomy question, sampled or not
        print("Anatomy Identification questions:", len(anat), "-> sampling", min(anat_size, len(anat)))
        if len(anat) > anat_size:  # open-access has ~16k; MedSTP reports 3,329 (plan.md §C2)
            anat = sorted(random.Random(seed).sample(anat, anat_size), key=lambda s: s.qid)

        rng = random.Random(seed)
        panel: list[Sample] = []
        for name in PANEL_MODALITIES:
            pool = [it for it in items if normalize_modality(it["modality_type"]) == name
                    and f"omni/{it['question_id']}" not in anat_ids]
            rng.shuffle(pool)
            per_image: Counter = Counter()
            picked = []
            for it in pool:
                if per_image[it["image_path"]] >= max_per_image:
                    continue
                s = to_sample(it, "omni_panel")
                if s is None:
                    continue
                picked.append(s)
                per_image[it["image_path"]] += 1
                if len(picked) == panel_per_modality:
                    break
            if len(picked) < panel_per_modality:
                print(f"warning: only {len(picked)} questions available for {name}")
            panel.extend(picked)

        for s in anat + panel:
            src = prefix + s.image_path.removeprefix("images/omnimedvqa/")
            dst = Path(out_root) / s.image_path
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(zf.read(src))
    write_manifest(anat, out_root, "omni_anat")
    write_manifest(panel, out_root, "omni_panel")
    return {"omni_anat": len(anat), "omni_panel": len(panel)}


# ------------------------------------------------------------------------------------------ RadImageNet

def _strip_option_tag(text: str) -> str:
    return re.sub(r"^\s*\(?[A-Da-d][\.\):]\s*", "", str(text)).strip()


def build_radimagenet(parquet_path: Path, out_root: Path,
                     datasets: tuple[str, ...] | list[str] | None = None) -> dict[str, int]:
    """RadImageNet-VQA benchmark → rin_anat_mc and rin_abn_closed, one question per image each (plan.md §C2).

    Run inside a GPU session only (license forbids redistribution). Prints the field values it filtered on so the
    choice can be recorded; raises if a subset comes out empty.
    """
    import pyarrow.parquet as pq

    df = pq.read_table(parquet_path).to_pandas()
    missing = {"image", "question", "choices", "answer", "question_type", "metadata"} - set(df.columns)
    if missing:
        raise KeyError(f"RadImageNet parquet lacks columns {missing}; has {list(df.columns)}")
    meta = df["metadata"].apply(lambda m: m if isinstance(m, dict) else (json.loads(m) if isinstance(m, str) else {}))
    content = meta.apply(lambda m: str(m.get("content_type", "")).lower())
    qtype = df["question_type"].astype(str).str.lower()
    print("question_type values:", dict(Counter(qtype)))
    print("metadata.content_type values:", dict(Counter(content)))
    print("metadata.question_id values:", dict(Counter(meta.apply(lambda m: str(m.get("question_id", ""))))))

    def image_key(img) -> tuple[str, bytes]:
        data = img["bytes"] if isinstance(img, dict) else img
        return hashlib.sha1(data).hexdigest(), data

    subsets = {
        "rin_anat_mc": (content.str.contains("anatom")) & qtype.str.contains("mc|multiple|choice"),
        "rin_abn_closed": (content.str.contains("abnormal")) & qtype.str.contains("closed|yes|binary"),
    }
    if datasets is not None:
        unknown = set(datasets) - set(subsets)
        if unknown:
            raise ValueError(f"Unknown RadImageNet datasets: {sorted(unknown)}")
        subsets = {name: subsets[name] for name in datasets}
    counts = {}
    for name, selector in subsets.items():
        samples, seen = [], set()
        for idx in np.flatnonzero(selector.to_numpy()):
            row, m = df.iloc[idx], meta.iloc[idx]
            key, data = image_key(row["image"])
            if key in seen:
                continue
            path = Path(out_root) / "images" / "radimagenet" / f"{key}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            # question_id names a template (e.g. anatomy_mc), not an individual image/question.
            common = dict(qid=f"{name}/{key}/{m.get('question_id', idx)}", dataset=name, image_id=f"rin/{key}",
                          image_path=f"images/radimagenet/{key}.png", question=str(row["question"]),
                          modality=str(m.get("modality", "")),
                          meta={k: str(v) for k, v in m.items()})
            if name == "rin_anat_mc":
                options = [_strip_option_tag(c) for c in list(row["choices"])]
                answer = str(row["answer"]).strip()
                if len(answer) == 1 and answer.upper() in "ABCD":
                    letter = answer.upper()
                    choices = tuple(f"{chr(65 + i)}. {t}" for i, t in enumerate(options))
                else:
                    choices, letter = _letter_choices(options, _strip_option_tag(answer))
                if letter is None:
                    continue
                samples.append(Sample(answer=letter, answer_type="mc", prompt_style="mc", choices=choices, **common))
            else:
                answer = str(row["answer"]).strip().lower()
                if answer not in ("yes", "no"):
                    continue
                samples.append(Sample(answer=answer, answer_type="yesno", prompt_style="judgement", **common))
            seen.add(key)
        if not samples:
            raise ValueError(f"{name}: no rows matched; adjust the filters using the printed values")
        write_manifest(samples, out_root, name)
        counts[name] = len(samples)
    return counts


# ------------------------------------------------------------------------------------- natural images

def build_natural(gqa_instructions: Path, gqa_images: Path, pope: Path, out_root: Path, n_each: int = 500,
                  seed: int = 0) -> dict[str, int]:
    """500 GQA testdev-balanced + 500 POPE adversarial questions (plan.md §C2)."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    gqa = pq.read_table(gqa_instructions).to_pandas()
    gqa = gqa.iloc[np.sort(rng.choice(len(gqa), size=n_each, replace=False))]
    images = pq.read_table(gqa_images).to_pandas().set_index("id")
    samples = []
    for _, r in gqa.iterrows():
        img = images.loc[r["imageId"], "image"]
        path = Path(out_root) / "images" / "gqa" / f"{r['imageId']}.jpg"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(path, quality=95)
        samples.append(Sample(qid=f"gqa/{r['id']}", dataset="natural", image_id=f"gqa/{r['imageId']}",
                              image_path=f"images/gqa/{r['imageId']}.jpg", question=str(r["question"]),
                              answer=str(r["answer"]), answer_type="closed", prompt_style="close", modality="natural",
                              meta={"source": "gqa"}))
    pope_df = pq.read_table(pope).to_pandas()
    pope_df = pope_df.iloc[np.sort(rng.choice(len(pope_df), size=n_each, replace=False))]
    for _, r in pope_df.iterrows():
        name = Path(str(r["image_source"])).stem
        path = Path(out_root) / "images" / "pope" / f"{name}.jpg"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(path, quality=95)
        samples.append(Sample(qid=f"pope/{r['question_id']}", dataset="natural", image_id=f"pope/{name}",
                              image_path=f"images/pope/{name}.jpg", question=str(r["question"]),
                              answer=str(r["answer"]).strip().lower(), answer_type="yesno", prompt_style="judgement",
                              modality="natural", meta={"source": "pope", "category": str(r.get("category", ""))}))
    write_manifest(samples, out_root, "natural")
    return {"natural": len(samples)}


# ------------------------------------------------------------------------------------------- masks

def build_masks(bundle: Path, datasets: list[str], models: list[str], overlays_per_modality: int = 20) -> None:
    """Per-token foreground fractions for every image of every dataset and model → masks/<dataset>__<model>.npz."""
    from .masks import MaskStore, image_gate, oracle_gate

    bundle = Path(bundle)
    (bundle / "masks").mkdir(exist_ok=True)
    for dataset in datasets:
        images = {}
        for s in load_manifest(bundle, dataset):
            images.setdefault(s.image_id, s)
        for model in models:
            ids, grids, fracs, oracle = [], [], [], []
            for image_id, s in images.items():
                img = Image.open(bundle / s.image_path)
                grid, frac = image_gate(img, model)
                ids.append(image_id)
                grids.append(grid)
                fracs.append(frac)
                mask_png = (bundle / s.image_path).with_name("mask.png")
                if dataset.startswith("slake") and mask_png.exists():
                    oracle.append(oracle_gate(img, Image.open(mask_png), model))
            MaskStore(ids, grids, fracs).save(bundle / "masks" / f"{dataset}__{model}.npz")
            if oracle and len(oracle) == len(ids):
                MaskStore(ids, grids, oracle).save(bundle / "masks" / f"{dataset}__{model}__oracle.npz")
            print(f"masks: {dataset} x {model}: {len(ids)} images")
        _write_overlays(bundle, images, models[0], overlays_per_modality)


def _write_overlays(bundle: Path, images: dict, model: str, per_modality: int) -> None:
    from .masks import MODEL_VIEWS, image_gate, model_view

    by_modality: dict[str, list] = defaultdict(list)
    for s in images.values():
        by_modality[s.modality].append(s)
    cell = MODEL_VIEWS[model]["cell"]
    for modality, samples in by_modality.items():
        out_dir = bundle / "masks" / "overlays" / re.sub(r"[^A-Za-z0-9_-]+", "_", modality)
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in random.Random(0).sample(samples, min(per_modality, len(samples))):
            img = Image.open(bundle / s.image_path)
            view, _ = model_view(img, model)
            _, frac = image_gate(img, model)
            arr = np.asarray(view).astype(np.float32)
            bg = np.kron((frac < 0.5).astype(np.float32), np.ones((cell, cell)))[..., None]
            shaded = arr * (1 - 0.6 * bg) + np.array([255, 0, 0]) * 0.6 * bg
            Image.fromarray(shaded.clip(0, 255).astype(np.uint8)).save(out_dir / f"{s.image_id.replace('/', '_')}.png")


# --------------------------------------------------------------------------------------- summary + CLI

def write_summary(bundle: Path) -> dict:
    """bundle_manifest.json: counts, background fractions and manifest hashes (copy into configs/)."""
    from .masks import MaskStore, background_fraction

    bundle = Path(bundle)
    summary = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": {}}
    try:
        summary["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        summary["git"] = ""
    for path in sorted((bundle / "manifests").glob("*.jsonl")):
        dataset = path.stem
        samples = load_manifest(bundle, dataset)
        entry = {
            "questions": len(samples),
            "images": len({s.image_id for s in samples}),
            "by_modality": dict(Counter(s.modality for s in samples)),
            "by_answer_type": dict(Counter(s.answer_type for s in samples)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "median_background_fraction": {},
        }
        for mask_path in sorted((bundle / "masks").glob(f"{dataset}__*.npz")):
            if mask_path.stem.endswith("__oracle"):
                continue
            store, model = MaskStore.load(mask_path), mask_path.stem.split("__")[1]
            per_modality: dict[str, list] = defaultdict(list)
            for s in {s.image_id: s for s in samples}.values():
                per_modality[s.modality].append(background_fraction(store.get(s.image_id)[1]))
            entry["median_background_fraction"][model] = {m: round(float(np.median(v)), 3)
                                                          for m, v in per_modality.items()}
        summary["datasets"][dataset] = entry
    (bundle / "bundle_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _download(repo: str, filename: str, work: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=work / repo.split("/")[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "masks", "summary"])
    parser.add_argument("--out", "--bundle", dest="bundle", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("data_work"), help="download/scratch directory")
    parser.add_argument("--only", default="slake,omnimedvqa,natural", help="comma list: slake,omnimedvqa,natural,radimagenet")
    parser.add_argument("--skip", default="", help="comma list of sources to skip")
    parser.add_argument("--models", default="lingshu7b,qwen3vl8b,huatuo7b")
    parser.add_argument("--datasets", default="", help="masks: comma list of datasets (default: every manifest)")
    args = parser.parse_args()
    bundle = args.bundle
    bundle.mkdir(parents=True, exist_ok=True)

    if args.command == "build":
        sources = [s for s in args.only.split(",") if s and s not in args.skip.split(",")]
        counts = {}
        if "slake" in sources:
            zip_path = _download("BoKelvin/SLAKE", "imgs.zip", args.work)
            test_json = _download("BoKelvin/SLAKE", "test.json", args.work)
            raw = test_json.parent
            if not (raw / "imgs").exists():
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(raw)
            shutil.copy(_download("BoKelvin/SLAKE", "mask.txt", args.work), bundle / "slake_mask_labels.txt")
            counts.update(build_slake(raw, bundle))
        if "omnimedvqa" in sources:
            counts.update(build_omnimedvqa(_download("foreverbeliever/OmniMedVQA", "OmniMedVQA.zip", args.work), bundle))
        if "natural" in sources:
            counts.update(build_natural(
                _download("lmms-lab/GQA", "testdev_balanced_instructions/testdev-00000-of-00001.parquet", args.work),
                _download("lmms-lab/GQA", "testdev_balanced_images/testdev-00000-of-00001.parquet", args.work),
                _download("lmms-lab/POPE", "Full/adversarial-00000-of-00001.parquet", args.work),
                bundle))
        if "radimagenet" in sources:
            counts.update(build_radimagenet(
                _download("raidium/RadImageNet-VQA", "benchmark/test-00000-of-00001.parquet", args.work), bundle,
                datasets=args.datasets.split(",") if args.datasets else None))
        print(json.dumps(counts, indent=2))
    elif args.command == "masks":
        datasets = [d for d in args.datasets.split(",") if d] or \
            sorted(p.stem for p in (bundle / "manifests").glob("*.jsonl"))
        build_masks(bundle, datasets, args.models.split(","))
    else:
        print(json.dumps(write_summary(bundle), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
