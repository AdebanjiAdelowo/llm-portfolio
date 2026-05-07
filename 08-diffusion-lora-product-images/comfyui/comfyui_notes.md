# ComfyUI Integration — Product Image Generation Workflow

## Portfolio Value

Adding ComfyUI to this project demonstrates:
- Practical deployment awareness (not just training scripts)
- Ability to deliver tooling non-technical stakeholders can use
- Familiarity with the node-based diffusion ecosystem used heavily in industry

**For a recruiter:** A ComfyUI workflow shows you can take a model checkpoint and wire it into a usable generation interface — a realistic output of an AI engineering role.

---

## Recommended Workflow: Product Variation Generator

**Goal:** Load the fine-tuned LoRA alongside the base SDXL model and generate consistent product image variations from a single text prompt.

**Use case a recruiter understands:**  
"A brand could describe a new colorway in text — `sks eyewear in matte silver frames` — and generate product photography variations for review without a photoshoot."

### Workflow Nodes

```
[Load Checkpoint (SDXL base)]
         │
[Load LoRA] ← outputs/lora_weights/pytorch_lora_weights.safetensors
         │
[CLIP Text Encode (positive)]  ←  prompt
[CLIP Text Encode (negative)]  ←  negative prompt
         │
[KSampler]
  steps: 30  |  cfg: 7.5  |  sampler: dpmpp_2m  |  scheduler: karras
         │
[VAE Decode]
         │
[Save Image] → results/comfyui_output/
```

### Key Node Settings

| Node | Setting | Value |
|---|---|---|
| Load Checkpoint | model | `sd_xl_base_1.0.safetensors` |
| Load LoRA | lora path | `lora_weights.safetensors` |
| Load LoRA | strength_model | 0.8 (tune: lower = subtler style shift) |
| Load LoRA | strength_clip | 0.8 |
| KSampler | steps | 30 |
| KSampler | cfg | 7.5 |
| KSampler | sampler | dpmpp_2m |
| Empty Latent Image | width × height | 1024 × 1024 |

**LoRA strength tuning:**  
- 0.6–0.8: Recommended starting range. Strong enough to activate style without over-constraining composition.  
- 1.0: Maximum domain adherence; may reduce diversity.  
- 0.4: Subtle influence; useful for blending base style with learned style.

---

## Setup Instructions

1. Install ComfyUI (one-time):
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI
   cd ComfyUI && pip install -r requirements.txt
   ```

2. Copy model and LoRA files:
   ```bash
   cp path/to/sd_xl_base_1.0.safetensors ComfyUI/models/checkpoints/
   cp outputs/lora_weights/pytorch_lora_weights.safetensors ComfyUI/models/loras/
   ```

3. Launch:
   ```bash
   python main.py --gpu-only
   ```

4. Import workflow:  
   Open ComfyUI in browser → Load → select `comfyui/workflow.json`

---

## Is Building a Custom Node Worth It?

**Short answer: No, not for this portfolio project.**

Custom nodes add complexity and maintenance burden without proportional portfolio signal unless the node solves a problem that doesn't exist in the ecosystem. For this project:

- The standard Load LoRA node handles your checkpoint correctly.
- The standard SDXL pipeline covers all needed inference steps.
- A clean, well-documented standard workflow is a better portfolio artifact than a half-finished custom node.

**When a custom node is worth it:**
- You need a node that wraps your curation pipeline (e.g., auto-caption on image upload)
- You're building a product and need a proprietary node for a non-standard model architecture
- You have 2+ weeks to implement, test, and document it properly

**For limited time:** document the standard workflow, export `workflow.json`, and write clear setup notes. That's the credible minimum.

---

## Documenting This for Recruiters

Include in your portfolio README or LinkedIn project post:
1. A 30-second screen recording of the ComfyUI workflow generating product variations
2. A before/after GIF or grid (base SDXL vs. LoRA-conditioned output in ComfyUI)
3. One sentence on what problem this solves: "Enables non-engineers to generate product photography variations from text descriptions using a fine-tuned model."

**What to say in an interview:**  
"I packaged the fine-tuned LoRA as a ComfyUI workflow so that a product or design team could generate variations without needing to write code. That's the deployment layer — knowing how to get a model checkpoint into a usable form."

---

## Workflow JSON

See `workflow.json` in this directory. Import it directly into ComfyUI.  
*(Export your actual workflow from ComfyUI after building it: Menu → Save → workflow.json)*
