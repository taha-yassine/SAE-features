import argparse
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset, Dataset
from tqdm import tqdm
from pathlib import Path
from importlib.resources import files
import yaml
import sys

def get_base_model(sae_release):
    with (Path(files("sae_lens")) / "pretrained_saes.yaml").open("r") as f:
        data = yaml.safe_load(f)
        try:
            model = data[sae_release]['model']
            return model
        except KeyError:
            print(f"Base model not found for {sae_release}.")
            sys.exit(1)

def activations_gen(model, tokenizer, sae, batched_dataset, device):
    for batch_idx, batch in enumerate(tqdm(batched_dataset, desc="Processing batches", unit="batch")):
        batch_size = len(batch['raw_content'])
        batch = tokenizer(batch['raw_content'], truncation=True, padding=True, return_tensors='pt').to(device)
        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], stop_at_layer=8)

        latents = sae.encode(outputs)
        latents = latents * batch['attention_mask'][:, :, torch.newaxis] # Zero out padding tokens

        # Find non-zero activations
        mask = latents.abs() > 1e-5
        locations = torch.nonzero(mask)
        activations = latents[mask]

        # Convert to lists and yield individual elements
        yield from ({"batch_idx": int(b)+batch_idx*batch_size, 
                    "ctx_pos": int(c), 
                    "feature_idx": int(f),
                    "activation": a} 
                   for b, c, f, a 
                   in torch.cat([locations, activations.unsqueeze(-1)], dim=-1).tolist())

        # Free up memory
        del batch
        del outputs
        del latents
        del mask
        del locations
        del activations
        torch.cuda.empty_cache()

def cache_activations(dataset_name, sae_release, sae_id, batch_size, output_dir, streaming=False):
    # Disable gradient calculation
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model and tokenizer
    model = HookedTransformer.from_pretrained(get_base_model(sae_release), device=device)
    tokenizer = model.tokenizer

    # Initialize SAE
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    # Load and batch the dataset
    # TODO: have a way to automatically determine correct column name
    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True, streaming=streaming).select_columns(['raw_content'])
    batched_dataset = dataset.batch(batch_size=batch_size)


    # Create dataset from generator
    activations_dataset = Dataset.from_generator(
        activations_gen,
        gen_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "sae": sae,
            "batched_dataset": batched_dataset,
            "device": device
        }
    )
    
    # TODO: add metadata

    # Save the dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache activations from a language model.")
    parser.add_argument("--dataset", default="Skylion007/openwebtext", help="Name of the dataset to use")
    parser.add_argument("--sae-release", default="gpt2-small-res-jb-feature-splitting", help="SAE release name")
    parser.add_argument("--sae-id", default="blocks.8.hook_resid_pre_768", help="SAE ID")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--output-dir", default=None, help="Output directory path")
    parser.add_argument("--streaming", action="store_true", help="Process dataset in streaming mode")

    args = parser.parse_args()

    args.output_dir = (
        Path(args.output_dir) 
        if args.output_dir is not None 
        else Path("cached_activations")
    ) / args.sae_release / args.dataset.split('/')[-1]

    cache_activations(
        args.dataset, 
        args.sae_release, 
        args.sae_id, 
        args.batch_size, 
        args.output_dir,
        args.streaming
    )
