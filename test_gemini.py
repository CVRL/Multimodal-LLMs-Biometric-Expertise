from __future__ import annotations
import sys
from PIL import Image
import glob
import os
import csv
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import json, random
from typing import Dict, List, Any, Optional
import pdb
import traceback

# --- Import the Google Gemini library ---
import google.generativeai as genai

def load_env_file(env_path=".env"):
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"No .env file found at {env_path}")
    
    with open(env_path, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            key, value = line.strip().split("=", 1)
            os.environ[key] = value

# --- Load and configure the Gemini API key ---
load_env_file()
api_key = os.getenv("GEMINI_API_KEY") # Switched to GEMINI_API_KEY
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY environment variable!")

genai.configure(api_key=api_key)

# ---------- Data ingestion & organization ----------
def load_iris_jsons(dir_path: str | Path) -> Dict[str, List[dict]]:
    """
    Read all *.json files in dir_path. Each file contains a list of records:
      {
        "irisImageLink": "image_name.png",
        "attackType": "Live" | "...",
        "humanExaminers": [{"identifier": "...", "correctlyClassified": bool, "verbalDescription": str}, ...]
      }

    Returns a dict keyed by attackType -> list of records.
    Attack type sits at the highest level for easy class-wise sampling.
    """
    dir_path = Path(dir_path)
    data_by_type: Dict[str, List[dict]] = defaultdict(list)

    for jf in sorted(dir_path.glob("*.json")):
        with jf.open("r", encoding="utf-8") as f:
            try:
                items = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {jf}: {e}") from e

        if not isinstance(items, list):
            raise ValueError(f"Expected a list in {jf}, got {type(items)}")

        for rec in items:
            # Defensive normalization
            attack = str(rec.get("attackType", "Unknown")).strip()
            entry = {
                "irisImageLink": rec.get("irisImageLink", ""),
                "attackType": attack,
                "humanExaminers": [],
                "Llama_MESH": rec.get("Llama_MESH",""),
                "Gemini_MESH": rec.get("Gemini_MESH","")
            }
            for ex in rec.get("humanExaminers", []):
                entry["humanExaminers"].append({
                    "identifier": ex.get("identifier", ""),
                    "correctlyClassified": bool(ex.get("correctlyClassified", False)),
                    # collapse newlines for prompt hygiene
                    "verbalDescription": str(ex.get("verbalDescription", "")).strip().replace("\n", " ")
                })
            data_by_type[attack].append(entry)

    # Convert defaultdict to normal dict for cleanliness
    return {k: v for k, v in data_by_type.items()}


# ---------- Sampling helpers ----------

def sample_one_record_per_type(
    data_by_type: Dict[str, List[dict]],
    seed: Optional[int] = None,
    max_examiners_per_type: Optional[int] = None
) -> Dict[str, dict]:
    """
    For each attack type, choose one random record (image) and return its examiners.
    Optionally cap the number of examiner descriptions per type.
    Result format:
      {
        "Live": {
           "irisImageLink": "...",
           "examiners": [{"identifier": "...", "correctlyClassified": bool, "verbalDescription": str}, ...]
        },
        "Print": { ... },
        ...
      }
    """
    rng = random.Random(seed)
    out: Dict[str, dict] = {}
    for attack_type, records in data_by_type.items():
        if not records:
            continue
        rec = rng.choice(records)
        ex_list = rec["humanExaminers"]
        if max_examiners_per_type is not None:
            ex_list = ex_list[:max_examiners_per_type]
        out[attack_type] = {
            "irisImageLink": rec.get("irisImageLink", ""),
            "examiners": ex_list,
            "Llama_MESH": rec.get("Llama_MESH",""),
            "Gemini_MESH": rec.get("Gemini_MESH","")
        }
    return out


# ---------- Prompt insertion ----------

def _examples_section_for_prompt(
    per_type_samples: Dict[str, dict],
    include_image_links: bool = False,
    title: str = "Here are class-specific examples with human descriptions",
    json_name: str = "examiners"
) -> str:
    """
    Build a readable, LLM-friendly examples block.
    Live always listed first (if present), others follow alphabetically.
    """
    def status_word(correct: bool) -> str:
        return "correctly" if correct else "incorrectly"

    ordered_types = []
    if "Live" in per_type_samples:
        ordered_types.append("Live")
    ordered_types += sorted(t for t in per_type_samples.keys() if t != "Live")

    lines: List[str] = []
    lines.append(title + " (used as in-context guidance):\n")

    for t in ordered_types:
        block = per_type_samples[t]
        header = "Live iris:" if t == "Live" else f"Attack type — {t}:"
        lines.append(header)
        if include_image_links and block.get("irisImageLink"):
            lines.append(f"- Image: {block['irisImageLink']}")
        if not block.get(json_name):
            lines.append("- (No examiner descriptions available)")
        else:
            if json_name == "examiners":
                for ex in block[json_name]:
                    sw = status_word(ex.get("correctlyClassified", False))
                    desc = ex.get("verbalDescription", "")
                    # Use a consistent bullet format friendly to LLMs
                    lines.append(f'- User {sw} identified — "{desc}"')
            else:
                ex = block[json_name]
                lines.append(f'- User correctly identified - "{ex}"')
        lines.append("")  # blank line between classes

    return "\n".join(lines).rstrip() + "\n\n"


def inject_examples_above_required_output(
    base_prompt: str,
    examples_section: str,
    anchor: str = "Required Output Format"
) -> str:
    """
    Insert examples just before the 'Required Output Format' section.
    If the anchor isn't found, append to the end.
    """
    idx = base_prompt.find(anchor)
    if idx == -1:
        return base_prompt.rstrip() + "\n\n" + examples_section
    return base_prompt[:idx].rstrip() + "\n\n" + examples_section + base_prompt[idx:]


def make_prompt_with_per_class_examples(
    base_prompt: str,
    data_by_type: Dict[str, List[dict]],
    seed: Optional[int] = random.randint(0,10000),
    max_examiners_per_type: Optional[int] = 2,
    include_image_links: bool = False,
    json_name: str = "examiners"
) -> str:
    """
    One-stop function:
      1) sample one record per attack type,
      2) build an examples section,
      3) inject it above 'Required Output Format'.
    """
    samples = sample_one_record_per_type(
        data_by_type, seed=seed, max_examiners_per_type=max_examiners_per_type
    )
    examples_section = _examples_section_for_prompt(
        samples, include_image_links=include_image_links,
        title="Here is an example of each attack type (and Live) with user descriptions",
        json_name = json_name
    )
    return inject_examples_above_required_output(base_prompt, examples_section)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-imageFolder', default='../../datasets/CYBORG_iris/', type=str)
    parser.add_argument('-csv', default="../../datasets/CYBORG_iris/csvs_gemini/test_gemini_sample_30.csv", type=str)
    parser.add_argument('-output_dir', default="./gemini_annotations/", type=str)
    parser.add_argument('-output_filename', default="gemini_artificial_shortprompt.csv", type=str)
    parser.add_argument('-prompttype', default='shortprompt_iris', type=str)
    parser.add_argument('-img_output_dir', default='./gemini_annotations/', type=str) # Changed from chatgpt
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    subdir = args.csv.split('/')[-1][:-4]
    final_save_loc = args.img_output_dir + subdir + '/'
    os.makedirs(args.img_output_dir, exist_ok=True)
    os.makedirs(final_save_loc, exist_ok=True)

    # --- Initialize the Gemini model ---
    # Using gemini-1.5-flash for speed and cost-effectiveness with vision tasks
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    # --- Define the prompt once ---
    if args.prompttype == "shortprompt_iris" or \
       args.prompttype == "shorthuman_iris" or \
       args.prompttype == "shortllamamesh_iris" or \
       args.prompttype == "shortgeminimesh_iris":
        base_prompt = 'Is this a real and healthy iris or synthetic/unhealthy iris? Return a single float number from 0 to 1 with 0 being real/healthy and 1 being synthetic/unhealthy and no other output. Always return only a single float number even if the image does not appear to be an iris.'

    if args.prompttype == "longprompt_iris" or \
       args.prompttype == "longhuman_iris" or \
       args.prompttype == "longgeminimesh_iris" or \
       args.prompttype == "longllamamesh_iris":
        base_prompt = "Task Overview\n"+\
                 "You are an expert iris biometric analyst tasked with evaluating iris samples for authentication validity. Using the knowledge base of attack patterns provided below, analyze the query iris image to determine if it represents a legitimate biometric sample or a presentation attack.\n"+\
                 "Classification Context\n"+\
                 "CRITICAL: Only healthy, live human eyes are considered legitimate for biometric authentication. All other categories must be classified as spoofs, including but not limited to:\n"+\
                 "Live eyes with diseases/medical conditions (miosis, mydriasis, cataracts, glaucoma, etc.)\n"+\
                 "Print attacks and display attacks\n"+\
                 "Synthetic generation (StyleGAN, etc.) and 3D models\n"+\
                 "Any pathological or abnormal eye conditions\n"+\
                 "Contact lens artifacts or prosthetic eyes\n"+\
                 "Analysis Framework\n"+\
                 "Examine the following key features systematically:\n"+\
                 "Iris texture and patterns: Crypts, furrows, pigment spots, radial fibers\n"+\
                 "Pupil characteristics: Shape, boundary definition, size, concentricity\n"+\
                 "Corneal reflections: Position, intensity, consistency with lighting\n"+\
                 "Eyelash and periocular region: Natural appearance, shadowing\n"+\
                 "Image artifacts: Compression, pixelation, unnatural edges\n"+\
                 "Lighting consistency: Shadows, highlights, color temperature\n"+\
                 "Medical/pathological indicators: Structural abnormalities, disease markers\n"+\
                 "Spoofing indicators: Print quality, display characteristics, synthetic artifacts\n"+\
                 "Required Output Format\n"+\
                 "Label: [Live or Spoof]\n"+\
                 "Confidence: [0.0 - 1.0]\n"+\
                 "Explanation: [4-5 sentences describing the image and key features that led to the classification decision, including specific anatomical observations and technical indicators]\n"+\
                 "Analysis Instructions\n"+\
                 "Using the knowledge base of attack patterns provided above, carefully analyze the query iris image. Consider how the observed features compare to the documented attack signatures and legitimate iris characteristics. Focus on the technical and anatomical details most relevant to biometric authentication validity. Provide specific, measurable observations that support your classification decision.\n"

    if args.prompttype == "contrastprompt_iris":
        base_prompt = 'Task Overview\n'+\
                'You are an expert iris biometric analyst tasked with evaluating iris samples for authentication validity. Using the knowledge base of attack patterns provided below, analyze the query iris image through a contrastive reasoning framework to determine if it represents a legitimate biometric sample or a presentation attack.\n'+\
                'Classification Context\n'+\
                'CRITICAL: Only healthy, live human eyes are considered legitimate for biometric authentication. All other categories must be classified as spoofs, including but not limited to:\n'+\
                'Live eyes with diseases/medical conditions (miosis, mydriasis, cataracts, glaucoma, etc.)\n'+\
                'Print attacks and display attacks\n'+\
                'Synthetic generation (StyleGAN, etc.) and 3D models\n'+\
                'Any pathological or abnormal eye conditions\n'+\
                'Contact lens artifacts or prosthetic eyes\n'+\
                'Analysis Framework\n'+\
                'Examine the following key features systematically:\n'+\
                'Iris texture and patterns: Crypts, furrows, pigment spots, radial fibers\n'+\
                'Pupil characteristics: Shape, boundary definition, size, concentricity\n'+\
                'Corneal reflections: Position, intensity, consistency with lighting\n'+\
                'Eyelash and periocular region: Natural appearance, shadowing\n'+\
                'Image artifacts: Compression, pixelation, unnatural edges\n'+\
                'Lighting consistency: Shadows, highlights, color temperature\n'+\
                'Medical/pathological indicators: Structural abnormalities, disease markers\n'+\
                'Spoofing indicators: Print quality, display characteristics, synthetic artifacts\n'+\
                'Required Output Format\n'+\
                'Evidence for "Live" Classification:\n'+\
                '[4-5 sentences describing specific anatomical features, texture characteristics, and technical indicators that support the sample being a genuine, healthy live iris. Include concrete observations about iris structure, natural variations, and biometric authenticity markers.]\n'+\
                'Evidence for "Spoof" Classification:\n'+\
                '[4-5 sentences describing specific anomalies, artifacts, inconsistencies, or suspicious characteristics that suggest presentation attack or non-authentic sample. Include concrete observations about potential attack vectors, abnormalities, or technical irregularities.]\n'+\
                'Final Judgment:\n'+\
                'Label: [Live or Spoof]\n'+\
                'Confidence: [0.0 - 1.0]\n'+\
                'Justification: [4-5 sentences providing the definitive reasoning for the chosen classification, explaining why the evidence for one label outweighs the evidence for the other, and highlighting the most discriminative features that determined the final decision.]\n'+\
                'Analysis Instructions\n'+\
                'Using the knowledge base of attack patterns provided above, carefully analyze the query iris image from both perspectives. First, identify all features that could support a "Live" classification, considering natural anatomical characteristics and authentic biometric markers. Then, identify all features that could indicate a "Spoof" classification, considering known attack patterns and anomalies. Finally, weigh the competing evidence to reach a definitive conclusion, clearly articulating which indicators are most decisive and why they tip the balance toward your final classification.'

    imageCSV = open(args.csv, "r")
    with open(f'{args.output_dir}/{args.output_filename}', 'w') as outF:
        for entry in tqdm(imageCSV):
            tokens = entry.split(",")
            if tokens[0] != 'test':
                continue
            
            label = tokens[1]
            
            
            upd_name = tokens[-2].replace("\n", "")
            imgFile = f'{args.imageFolder}{upd_name}'

            try:
                if "human_text" in args.prompttype:
                    data_by_type = load_iris_jsons("../../datasets/LLM_Stuff/JSON/Batch_Info/")
                    per_type_samples = sample_one_record_per_type(data_by_type, max_examiners_per_type=None)
                    prompt = make_prompt_with_per_class_examples(base_prompt, data_by_type, max_examiners_per_type=None, include_image_links=False,json_name="examiners")
                    print(prompt)
                elif "shorthuman_iris" == args.prompttype:
                    data_by_type = load_iris_jsons("../../datasets/Verbal_Descriptions/MESH_Batches")
                    per_type_samples = sample_one_record_per_type(data_by_type, max_examiners_per_type=None)
                    prompt = make_prompt_with_per_class_examples(base_prompt, data_by_type, max_examiners_per_type=None, include_image_links=False,json_name="examiners")
                    print(prompt)
                elif "shortllamamesh_iris" == args.prompttype:
                    data_by_type = load_iris_jsons("../../datasets/Verbal_Descriptions/MESH_Batches")
                    per_type_samples = sample_one_record_per_type(data_by_type, max_examiners_per_type=None)
                    prompt = make_prompt_with_per_class_examples(base_prompt, data_by_type, max_examiners_per_type=None, include_image_links=False,json_name="Llama_MESH")
                    print(prompt)
                elif "longllamamesh_iris" == args.prompttype:
                    data_by_type = load_iris_jsons("../../datasets/Verbal_Descriptions/MESH_Batches")
                    per_type_samples = sample_one_record_per_type(data_by_type, max_examiners_per_type=None)
                    prompt = make_prompt_with_per_class_examples(base_prompt, data_by_type, max_examiners_per_type=None, include_image_links=False,json_name="Llama_MESH")
                    print(prompt)
                elif "shortgeminimesh_iris" == args.prompttype:
                    data_by_type = load_iris_jsons("../../datasets/Verbal_Descriptions/MESH_Batches")
                    per_type_samples = sample_one_record_per_type(data_by_type, max_examiners_per_type=None)
                    prompt = make_prompt_with_per_class_examples(base_prompt, data_by_type, max_examiners_per_type=None, include_image_links=False,json_name="Gemini_MESH")
                    print(prompt)
                elif "longgeminimesh_iris" == args.prompttype:
                    data_by_type = load_iris_jsons("../../datasets/Verbal_Descriptions/MESH_Batches")
                    per_type_samples = sample_one_record_per_type(data_by_type, max_examiners_per_type=None)
                    prompt = make_prompt_with_per_class_examples(base_prompt, data_by_type, max_examiners_per_type=None, include_image_links=False,json_name="Gemini_MESH")
                    print(prompt)
                else:
                    prompt = base_prompt

                # --- Gemini handles PIL Image objects directly, no base64 encoding needed ---
                img = Image.open(imgFile)
                
                # --- Simpler API call for Gemini ---
                # Pass a list containing the text prompt and the image object
                response = model.generate_content([prompt, img])
                print(response) 
                # --- Write the response text to the output file ---
                outF.write(f'{label}--{imgFile},{response.text.strip()}\n')
                outF.flush()

            except FileNotFoundError:
                print(f"Warning: Image file not found at {imgFile}. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing {imgFile}: {e}\n{print(traceback.format_exc())}")
