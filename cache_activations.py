import argparse
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm
    
def cache_activations(model_name, dataset_name, sae_release, sae_id, batch_size, output_file):
    # Disable gradient calculation
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer

    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
    batched_dataset = dataset.batch(batch_size=batch_size)

    idx = []
    activations = []

    for batch in tqdm(batched_dataset, desc="Processing batches", unit="batch"):
        batch = tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt').to(device)
        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], stop_at_layer=8)

        latents = sae.encode(outputs)
        latents = latents * batch['attention_mask'][:, :, torch.newaxis] # Zero out padding tokens

        idx.append(torch.nonzero(latents.abs() > 1e-5))
        activations.append(latents[latents.abs() > 1e-5])

        del batch
        del outputs
        del latents
        torch.cuda.empty_cache()

    idx = torch.cat(idx)
    activations = torch.cat(activations)

    # Save to file
    tensors_dict = {
        "locations": idx,
        "activations": activations
    }
    metadata = {
        "model": model_name,
        "dataset": dataset_name,
        "sae_release": sae_release,
        "sae_id": sae_id
    }

    save_file(tensors_dict, output_file, metadata=metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache activations from a language model.")
    parser.add_argument("--model", default="gpt2", help="Name of the language model to use")
    parser.add_argument("--dataset", default="Skylion007/openwebtext", help="Name of the dataset to use")
    parser.add_argument("--sae-release", default="gpt2-small-res-jb-feature-splitting", help="SAE release name")
    parser.add_argument("--sae-id", default="blocks.8.hook_resid_pre_768", help="SAE ID")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--output", default=None, help="Output file name")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"cached_activations/{args.sae_release}-{args.dataset.split('/')[-1]}.safetensors"

    cache_activations(args.model, args.dataset, args.sae_release, args.sae_id, args.batch_size, args.output)
