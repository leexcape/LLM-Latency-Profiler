import argparse
import torch
import time
import json
import platform
import subprocess
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_hardware_info(device):
    info = {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform()
    }

    # Try to get board model (for Raspberry Pi, Jetson, etc.)
    model_path = "/proc/device-tree/model"
    if os.path.isfile(model_path):
        try:
            with open(model_path) as f:
                info["hardware_model"] = f.read().strip()
        except Exception:
            info["hardware_model"] = "unreadable"

    # Try to get Jetson version info
    jetson_release = "/etc/nv_tegra_release"
    if os.path.isfile(jetson_release):
        try:
            with open(jetson_release) as f:
                info["jetson_info"] = f.read().strip()
        except Exception:
            info["jetson_info"] = "unreadable"

    # If CUDA is available, add device info
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(device)
            info.update({
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_MB": props.total_memory // (1024 * 1024),
            })
        except Exception:
            info["cuda_error"] = "Unable to get torch.cuda properties"

        # Try to run nvidia-smi
        try:
            smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"], encoding="utf-8")
            info["nvidia_smi"] = smi_output.strip()
        except Exception:
            info["nvidia_smi"] = "nvidia-smi not available or failed"

    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type (e.g., float32, float16, bfloat16)")
    args = parser.parse_args()

    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, args.dtype)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch_dtype)
    model.eval()

    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt")

    # Generate
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
    end_time = time.time()

    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Stats
    elapsed = end_time - start_time
    tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_sec = tokens_generated / elapsed

    # Collect metadata
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model_name": args.model,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "response_length": args.max_new_tokens,
        "elapsed_time_sec": elapsed,
        "tokens_per_second": tokens_per_sec,
        "dtype": args.dtype,
        "device_info": get_hardware_info(device)
    }

    # Save JSON
    out_name = f"results/llm_benchmark_{results['timestamp']}.json"
    with open(out_name, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBenchmark saved to {out_name}")
    print(f"\nGenerated Text:\n{'='*60}\n{generated_text}\n{'='*60}")


if __name__ == "__main__":
    main()
