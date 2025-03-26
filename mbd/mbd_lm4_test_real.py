#!/usr/bin/env python3
"""
Train and evaluate MBD-S on WikiText-2 with enhanced feedback, validation,
and integrated diffusion-based generation sampling.
"""

import argparse
import logging
import math
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

# --- Import necessary components from your mbd_lm4.py file ---
# Assuming mbd_lm4.py is in the same directory or accessible via PYTHONPATH
try:
    from mbd_lm4 import (
        MBDSFinal, MBDConfig, TierConfig, TIER_CONFIGS_EVOLVED,
        setup_dataloader, count_parameters,
        DEVICE, AMP_ENABLED, AMP_DTYPE, # Import necessary constants/functions
        # The global PAD_TOKEN_ID in mbd_lm4 might be used, we'll try to update it
    )
except ImportError as e:
    print(f"💥 Error importing from mbd_lm4.py: {e}")
    print("🚨 Please ensure mbd_lm4.py is in the Python path.")
    sys.exit(1)

# --- Dataset and Tokenizer Imports ---
try:
    from datasets import load_dataset, disable_caching
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    from tqdm import tqdm # For exciting progress bars!
    libs_available = True
    # Uncomment if caching causes issues, especially with multiprocessing
    # disable_caching()
except ImportError:
    print("💥 Error: `datasets`, `transformers`, and `tqdm` libraries are required. Install with:")
    print("pip install datasets transformers tqdm")
    libs_available = False
    sys.exit(1)

# --- Logging Setup ---
# Emojis make logging more fun! ✨
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ✨ %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Global variable for PAD_TOKEN_ID (will be set after tokenizer load) ---
_GLOBAL_PAD_TOKEN_ID: Optional[int] = None

# --- WikiText Processing ---

class WikiTextDataset(Dataset):
    """Dataset class for processing WikiText into fixed-length sequences."""
    def __init__(self, tokenized_ids: list, seq_len: int):
        self.seq_len = seq_len
        num_tokens = len(tokenized_ids)
        # Drop the last partial sequence to ensure all sequences are full length
        num_sequences = num_tokens // seq_len
        self.data = tokenized_ids[:num_sequences * seq_len]
        # Use f-strings for formatted output
        logging.info(f"📊 Created dataset with {num_sequences:,} sequences of length {seq_len}")

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        # Return sequence as a tensor
        return torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)

def prepare_wikitext_data(dataset_name: str, tokenizer: PreTrainedTokenizerBase, seq_len: int, num_proc: Optional[int] = 1) -> Dict[str, Dataset]:
    """Loads, tokenizes, and prepares WikiText data."""
    logging.info(f"📚 Loading dataset: {dataset_name}")
    try:
        raw_datasets = load_dataset("Salesforce/wikitext", dataset_name, trust_remote_code=True)

        raw_datasets['train'] = load_dataset("Salesforce/wikitext", dataset_name, split='train[:10%]')
        raw_datasets['validation'] = load_dataset("Salesforce/wikitext", dataset_name, split='validation[:20%]')  # Maybe a bit more validation

        #raw_datasets = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
        #raw_datasets = load_dataset("ptb_text_only", "penn_treebank", trust_remote_code=True)


    except Exception as e:
        logging.error(f"💥 Failed to load dataset '{dataset_name}'. Error: {e}")
        logging.error("Ensure you have internet connection and the dataset name is correct.")
        sys.exit(1)

    # Filter out empty lines which are common in WikiText-raw
    def filter_non_empty(example):
        return len(example['text'].strip()) > 0

    for split in raw_datasets:
        original_len = len(raw_datasets[split])
        raw_datasets[split] = raw_datasets[split].filter(filter_non_empty, num_proc=num_proc)
        filtered_len = len(raw_datasets[split])
        if original_len > filtered_len:
             logging.info(f"🧹 Filtered out {original_len - filtered_len} empty lines from '{split}' split.")

    # --- Tokenization ---
    def tokenize_function(examples):
        # Tokenize without adding special tokens automatically within segments
        return tokenizer(examples["text"], add_special_tokens=False)

    logging.info(f"⚙️ Tokenizing dataset using {num_proc} processes... (This might take a moment)")
    try:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing splits", # Description for progress bar
            num_proc=num_proc,
        )
    except Exception as e:
        logging.error(f"💥 Tokenization failed. Error: {e}")
        logging.error("If using multiple processes (`--num_proc > 1`), try setting TOKENIZERS_PARALLELISM=false environment variable or use --num_proc=1.")
        sys.exit(1)

    # --- Concatenate and Chunk ---
    datasets = {}
    for split in tokenized_datasets:
        logging.info(f"🧩 Processing split: {split}")
        # Concatenate all token IDs efficiently
        # Ensure input_ids exist and handle potential variations in dataset structure
        if 'input_ids' not in tokenized_datasets[split].column_names:
            logging.error(f"💥 'input_ids' column not found in tokenized dataset for split '{split}'. Columns: {tokenized_datasets[split].column_names}")
            sys.exit(1)

        concatenated_ids = [id for input_ids_list in tokenized_datasets[split]['input_ids'] for id in input_ids_list]
        logging.info(f"  Total tokens in '{split}': {len(concatenated_ids):,}")
        datasets[split] = WikiTextDataset(concatenated_ids, seq_len)

    return datasets

# --- Evaluation Function for MBD-S (MSE Loss) ---

@torch.no_grad() # Essential for evaluation mode
def evaluate_mbds_loss(model: MBDSFinal, data_loader: DataLoader, step: int) -> float:
    """Calculates the average validation MSE loss for the MBD-S model."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_batches = 0
    eval_start_time = time.time()
    logging.info(f"⏳ Starting validation at step {step}...")

    # Add tqdm for validation progress
    val_iterator = tqdm(data_loader, desc=f"Validation Step {step}", leave=False, dynamic_ncols=True)
    for batch in val_iterator:
        # MBD-S expects tensors directly
        if isinstance(batch, torch.Tensor):
             input_ids = batch.to(DEVICE, non_blocking=True)
        else:
             # Skip unexpected batch types gracefully
             logging.warning(f"🤔 Skipping unexpected batch type in validation: {type(batch)}")
             continue

        # Use autocast for consistency with training, though gradients aren't needed
        with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
            # Assume model returns (prediction, loss)
            # We only need the loss for evaluation here.
            _, loss = model(input_ids) # Targets default to input_ids

        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item()
            total_batches += 1
        else:
            logging.warning(f"⚠️ Invalid loss ({loss}) encountered during validation.")

    eval_time = time.time() - eval_start_time
    if total_batches == 0:
        logging.warning("🤷 No valid batches processed during validation.")
        # Do not set model back to train mode here, let the caller handle it
        return float('inf')

    avg_loss = total_loss / total_batches
    logging.info(f"✅ Validation Step {step} Complete! Avg MSE Loss: {avg_loss:.6f} (took {eval_time:.2f}s)")
    # Do not set model back to train mode here, let the caller handle it
    return avg_loss

# --- Generation using the model's internal method ---

@torch.no_grad()
def generate_sample(model: MBDSFinal, # Type hint specifically for MBDSFinal
                    tokenizer: PreTrainedTokenizerBase,
                    prompt: str,
                    max_new_tokens: int = 50,
                    ddim_steps: Optional[int] = None, # Add DDIM steps arg
                    temperature: float = 1.0, # Add temperature arg
                    device: torch.device = DEVICE # Pass device
                   ) -> Optional[str]:
    """
    Generates a text sample using the MBDSFinal model's built-in diffusion generation method.
    """
    # No need to check for lm_head, as we use model.generate()
    model.eval() # Ensure model is in evaluation mode
    logging.info(f"✍️ Generating sample using MBDSFinal.generate from prompt: '{prompt}'")
    logging.info(f"   (max_new: {max_new_tokens}, ddim_steps: {ddim_steps if ddim_steps else 'DDPM'}, temp: {temperature})")

    try:
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if input_ids.shape[1] == 0:
            logging.warning("🤔 Prompt resulted in empty input_ids. Cannot generate.")
            # model.train() - Caller should handle setting back to train mode
            return None

        # --- Call the model's own generate method ---
        # Ensure all necessary arguments for model.generate are provided
        # Assume model.generate handles device placement internally or expects inputs on correct device
        generated_ids = model.generate(
            prompt=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            ddim_steps=ddim_steps,
            # ddim_eta=0.0 # Use default eta from model.generate if needed
        )
        # --------------------------------------------

        # Decode the full generated sequence (model.generate likely handles EOS/padding)
        # generated_ids often includes the prompt part
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        logging.info(f"💡 Generated Text: '{generated_text}'")
        # Do not set model back to train mode here, let the caller handle it
        return generated_text

    except Exception as e:
        logging.error(f"💥 Error during MBDSFinal generation: {e}", exc_info=True)
        # Caller should handle setting back to train mode
        return None


# --- Main Training Script ---

def main(args):
    global _GLOBAL_PAD_TOKEN_ID # Allow modification
    if not libs_available:
        logging.error("💥 Required libraries not found. Exiting.")
        sys.exit(1)

    logging.info("🚀 Initializing MBD-S Training Run! Let's do this! 💪")
    logging.info(f"💾 Using device: {DEVICE}")
    if torch.cuda.is_available():
        logging.info(f"  GPU: {torch.cuda.get_device_name(DEVICE)}")
    if AMP_ENABLED:
        logging.info(f"⚡ Automatic Mixed Precision (AMP) enabled with dtype: {AMP_DTYPE}")
    else:
        logging.info("🐌 Automatic Mixed Precision (AMP) disabled (likely running on CPU).")

    # Set deterministic behavior if requested (for reproducibility, might impact performance)
    if args.seed is not None:
        logging.info(f"Setting random seed to {args.seed} for reproducibility.")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # Potentially enable deterministic algorithms, can slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Tokenizer ---
    logging.info(f"🔍 Loading tokenizer: {args.tokenizer_name}")
    try:
        # trust_remote_code=True might be needed for some tokenizers
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name) #, trust_remote_code=True)
    except Exception as e:
        logging.error(f"💥 Failed to load tokenizer '{args.tokenizer_name}'. Error: {e}")
        sys.exit(1)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            logging.warning(f"⚠️ Tokenizer '{args.tokenizer_name}' missing PAD token. Using EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}) as PAD.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Attempt to add a PAD token (requires resizing embeddings later)
            logging.warning(f"⚠️ Tokenizer '{args.tokenizer_name}' has no PAD token and no EOS token. Adding '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Model embedding layer will need resizing after initialization

    _GLOBAL_PAD_TOKEN_ID = tokenizer.pad_token_id
    # Attempt to update PAD_TOKEN_ID in the imported mbd_lm4 module
    try:
        import mbd_lm4
        if hasattr(mbd_lm4, 'PAD_TOKEN_ID'):
            mbd_lm4.PAD_TOKEN_ID = _GLOBAL_PAD_TOKEN_ID
            logging.info(f"Updated PAD_TOKEN_ID in mbd_lm4 module to {_GLOBAL_PAD_TOKEN_ID}")
        else:
            logging.warning("mbd_lm4 module does not have a PAD_TOKEN_ID attribute to update.")
    except (AttributeError, ImportError, NameError):
         logging.warning("Could not dynamically update PAD_TOKEN_ID in mbd_lm4. Ensure it uses the correct value internally.")

    vocab_size = len(tokenizer) # Recalculate if tokens were added
    logging.info(f"✅ Tokenizer loaded. Vocab size: {vocab_size}, PAD ID: {_GLOBAL_PAD_TOKEN_ID}")

    # --- Data ---
    effective_seq_len = args.seq_len
    logging.info(f"💾 Preparing data with sequence length: {effective_seq_len}")
    processed_datasets = prepare_wikitext_data(args.dataset_name, tokenizer, effective_seq_len, num_proc=args.num_proc)
    train_dataset = processed_datasets.get("train")
    val_dataset = processed_datasets.get("validation") or processed_datasets.get("test") # Fallback to test split

    if not train_dataset or not val_dataset:
        logging.error(f"💥 Could not load/process train or validation splits for {args.dataset_name}")
        sys.exit(1)

    logging.info(f"📦 Creating DataLoaders with batch size: {args.batch_size}")
    train_loader = setup_dataloader(train_dataset, args.batch_size, shuffle=True)
    # Use a potentially larger batch size for validation as no grads are needed
    val_batch_size = args.batch_size * 2
    val_loader = setup_dataloader(val_dataset, val_batch_size, shuffle=False)

    # --- Model ---
    logging.info(f"🛠️ Initializing MBD-S model with tier: {args.model_tier}")
    try:
        mbd_tier_dict = TIER_CONFIGS_EVOLVED[args.model_tier]
    except KeyError:
        logging.error(f"💥 Unknown model tier: {args.model_tier}. Available: {list(TIER_CONFIGS_EVOLVED.keys())}")
        sys.exit(1)

    mbd_tier_obj = TierConfig(**mbd_tier_dict)
    # Ensure pos_encoding_max_len is sufficient
    pos_encoding_max_len = max(args.seq_len + 10, 512) # A reasonable buffer/minimum
    mbd_config = MBDConfig(
        tier_config=mbd_tier_obj,
        vocab_size=vocab_size, # Use potentially updated vocab size
        embed_dim=args.embed_dim,
        l_prime=args.l_prime,
        width=1.0, # Standard width
        dropout=args.dropout,
        pos_encoding_max_len=pos_encoding_max_len,
        # pad_token_id = _GLOBAL_PAD_TOKEN_ID # Pass PAD ID explicitly if model's __init__ uses it
    )
    # Check if model's config takes pad_token_id, only pass if needed.
    # Based on provided mbd_lm4.py, MBDConfig doesn't take pad_token_id directly,
    # but the Embedding layer inside MBDSFinal uses the global one.

    try:
        model = MBDSFinal(mbd_config).to(DEVICE)
    except Exception as e:
        logging.error(f"💥 Failed to initialize MBDSFinal model. Error: {e}", exc_info=True)
        sys.exit(1)

    # Resize embeddings if a PAD token was added to the tokenizer AFTER vocab_size was initially determined
    # This assumes the original vocab_size was captured *before* adding the token.
    # A cleaner approach is to determine vocab_size definitively *after* tokenizer setup.
    if model.embedding.num_embeddings != vocab_size:
        logging.warning(f"Resizing model token embeddings from {model.embedding.num_embeddings} to {vocab_size} (likely due to added PAD token).")
        model.resize_token_embeddings(vocab_size)
        # Ensure embedding weights associated with PAD are zero if just resized
        with torch.no_grad():
             if _GLOBAL_PAD_TOKEN_ID is not None and 0 <= _GLOBAL_PAD_TOKEN_ID < model.embedding.num_embeddings:
                 model.embedding.weight[_GLOBAL_PAD_TOKEN_ID].zero_()
        # Update config if it stores vocab_size (MBDConfig doesn't seem to, but good practice)
        mbd_config.vocab_size = vocab_size


    num_params = count_parameters(model)
    logging.info(f"✅ MBD-S Model initialized ({args.model_tier} tier).")
    logging.info(f"   Parameters: {num_params:,}")
    # Safer attribute access using getattr
    logging.info(f"   Mamba Used: {getattr(getattr(model, 'mdb_block', None), 'use_mamba', 'N/A')}")
    logging.info(f"   Embed Dim: {getattr(model, 'embed_dim', 'N/A')}, L': {getattr(model, 'l_prime', 'N/A')}, Overlap: {getattr(model, 'overlap', 'N/A')}, Stride: {getattr(model, 'stride', 'N/A')}")
    logging.info(f"   Positional Encoding Max Len: {pos_encoding_max_len}")
    # Print model structure summary (optional, can be verbose)
    # logging.info(f"Model Structure:\n{model}")

    # --- Optimizer ---
    # Consider excluding bias/Norm layers from weight decay if needed (more advanced)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2))
    scaler = GradScaler(enabled=AMP_ENABLED) # For mixed precision

    # --- Training Loop ---
    logging.info(f"🔥 Starting training for {args.train_steps} steps!")
    start_time = time.time()
    train_losses = []
    validation_losses: Dict[int, float] = {} # Store validation loss at different steps
    best_val_loss = float('inf')
    best_val_step = -1
    steps_completed = 0
    # Use a data iterator to handle steps correctly across epochs implicitly
    train_iterator = iter(train_loader)
    best_model_saved_path = None

    # Main training loop using tqdm for progress
    pbar = tqdm(range(1, args.train_steps + 1), desc="Training Steps", dynamic_ncols=True)
    model.train() # Ensure model is in training mode

    try: # Wrap training loop in try/finally to ensure cleanup/summary
        for step in pbar:
            # --- Get Batch ---
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Epoch finished, restart iterator
                logging.info(f"🔄 Epoch finished after {step} steps, restarting data loader...")
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            # MBD-S expects tensors directly
            if isinstance(batch, torch.Tensor):
                 input_ids = batch.to(DEVICE, non_blocking=True)
            elif isinstance(batch, dict) and 'input_ids' in batch:
                 # Handle cases where dataloader might yield dicts
                 input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            else:
                 logging.warning(f"🤔 Skipping unexpected batch type in training: {type(batch)}")
                 continue

            # --- Training Step ---
            optimizer.zero_grad(set_to_none=True) # More efficient zeroing

            with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                # Assume model returns (prediction, loss)
                # Targets default to input_ids in MBDSFinal forward
                prediction, loss = model(input_ids)

                if loss is None or not torch.isfinite(loss):
                    logging.warning(f"⚠️ Invalid training loss ({loss}) at step {step}. Skipping batch gradient update.")
                    # Optionally: Add a counter for invalid losses and stop if too many occur
                    continue # Skip optimizer step and gradient update

            # --- Backpropagation ---
            scaler.scale(loss).backward()

            # --- Gradient Clipping (Optional but recommended) ---
            if args.grad_clip > 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                # Optional: Log grad norm if debugging
                # if step % (args.log_interval * 10) == 0:
                #     logging.debug(f"Step {step} Grad Norm: {grad_norm:.4f}")

            # --- Optimizer Step ---
            scaler.step(optimizer)
            scaler.update() # Update scaler for next iteration

            # --- Logging & Metrics ---
            loss_item = loss.item()
            train_losses.append(loss_item)
            steps_completed = step

            # Update progress bar description with dynamic best val loss
            pbar.set_postfix({"Train Loss": f"{loss_item:.4f}", "Best Val": f"{best_val_loss:.4f} @ {best_val_step}"})

            if step % args.log_interval == 0:
                # Calculate average loss over the logging interval for smoother reporting
                log_slice = train_losses[-args.log_interval:]
                avg_recent_loss = sum(log_slice) / len(log_slice) if log_slice else loss_item
                logging.info(f"   Step {step}/{args.train_steps} | Avg Train Loss ({args.log_interval} steps): {avg_recent_loss:.6f} | Current LR: {optimizer.param_groups[0]['lr']:.2e}")
                # Optional: Log GPU memory more sparsely
                if step % (args.log_interval * 10) == 0 and torch.cuda.is_available():
                     current_mem = torch.cuda.memory_allocated(DEVICE) / (1024**3) # GiB
                     peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**3) # GiB
                     logging.info(f"   GPU Memory Allocated: {current_mem:.2f} GiB | Peak: {peak_mem:.2f} GiB")

            # --- Periodic Validation & Generation ---
            if step % args.eval_interval == 0 or step == args.train_steps:
                val_loss = evaluate_mbds_loss(model, val_loader, step)
                # evaluate_mbds_loss leaves model in eval mode
                validation_losses[step] = val_loss

                # --- Try generating a sample using the model's method ---
                generated_text = None
                if args.generate_sample_prompt:
                     generated_text = generate_sample( # Call the updated function
                         model=model, # Already in eval mode
                         tokenizer=tokenizer,
                         prompt=args.generate_sample_prompt,
                         max_new_tokens=args.generation_max_new,
                         ddim_steps=args.generation_ddim_steps,
                         temperature=args.generation_temp,
                         device=DEVICE # Pass the device
                     )
                # ------------------------------------------------------

                # --- Checkpoint Saving (Best Model) ---
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    logging.info(f"🎉 New best validation loss: {val_loss:.6f} (Improved by {improvement:.4f} from {best_val_loss:.6f})")
                    best_val_loss = val_loss
                    best_val_step = step
                    if args.output_dir:
                        os.makedirs(args.output_dir, exist_ok=True)
                        best_save_path = os.path.join(args.output_dir, f"mbds_{args.model_tier}_wikitext2_best_val.pt")
                        logging.info(f"💾 Saving best model checkpoint to {best_save_path}")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'config': mbd_config, # Save model config
                            'train_args': vars(args), # Save training arguments
                            'step': step,
                            'best_val_loss': best_val_loss,
                            'tokenizer_name': args.tokenizer_name, # Save tokenizer info for reloading
                            'vocab_size': vocab_size,
                        }, best_save_path)
                        best_model_saved_path = best_save_path # Track path for final summary
                else:
                    logging.info(f"📉 Validation loss ({val_loss:.6f}) did not improve from best ({best_val_loss:.6f} @ step {best_val_step}).")

                # --- Set model back to training mode ---
                model.train()

    except KeyboardInterrupt:
        logging.warning("\n🛑 Training interrupted by user (Ctrl+C).")
    except Exception as e:
        logging.error(f"💥 An unexpected error occurred during training: {e}", exc_info=True)
    finally:
        # --- Training Finished / Interrupted ---
        training_time = time.time() - start_time
        logging.info(f"🏁 Training finished/stopped after {steps_completed} steps in {training_time:.2f} seconds ({training_time / 60:.2f} minutes).")

        final_train_loss = train_losses[-1] if train_losses else float('inf')
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float('inf')
        logging.info(f"   Final training step MSE loss: {final_train_loss:.6f}")
        logging.info(f"   Average training MSE loss: {avg_train_loss:.6f}")

        # --- Final Evaluation (if training completed some steps) ---
        final_val_loss = float('inf')
        if steps_completed > 0 and val_loader is not None:
            logging.info("⚙️ Performing final evaluation on validation set...")
            final_val_loss = evaluate_mbds_loss(model, val_loader, steps_completed)
            # Record final validation loss if not already done
            if steps_completed not in validation_losses:
                validation_losses[steps_completed] = final_val_loss
        else:
            logging.warning("Skipping final evaluation as no training steps were completed or validation loader missing.")


        logging.info("--- Training Summary ---")
        logging.info(f"📊 Dataset: {args.dataset_name}, Tokenizer: {args.tokenizer_name}")
        logging.info(f"🤖 Model Tier: {args.model_tier}, Parameters: {num_params:,}")
        logging.info(f"🔢 Training Steps Completed: {steps_completed}/{args.train_steps}")
        logging.info(f"📉 Final Training MSE Loss: {final_train_loss:.6f}")
        logging.info(f"📈 Final Validation MSE Loss: {final_val_loss:.6f}")
        logging.info(f"🏆 Best Validation MSE Loss: {best_val_loss:.6f} (achieved at step {best_val_step})")
        logging.info(f"⏱️ Total Training Time: {training_time:.2f} seconds")
        logging.info(f"💾 Best model saved to: {best_model_saved_path if best_model_saved_path else 'Not saved or N/A'}")

        # --- Optional: Save Final Model ---
        if args.output_dir and args.save_final_model and steps_completed > 0:
            final_save_path = os.path.join(args.output_dir, f"mbds_{args.model_tier}_wikitext2_final_step{steps_completed}.pt")
            logging.info(f"💾 Saving final model checkpoint to {final_save_path}")
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config': mbd_config,
                    'train_args': vars(args),
                    'step': steps_completed,
                    'final_train_loss': final_train_loss,
                    'final_validation_loss': final_val_loss,
                    'tokenizer_name': args.tokenizer_name,
                    'vocab_size': vocab_size,
                }, final_save_path)
            except Exception as e:
                logging.error(f"💥 Failed to save final model: {e}", exc_info=True)

        logging.info("🎉 Run complete! 🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Train MBD-S on WikiText-2 with Enhanced Feedback & Generation 🚀",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # Model args
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model_tier", type=str, default="core_balanced", choices=TIER_CONFIGS_EVOLVED.keys(), help="MBD-S tier configuration")
    model_group.add_argument("--embed_dim", type=int, default=512, help="Base embedding dimension")
    model_group.add_argument("--l_prime", type=int, default=64, help="Processing block length (L')") # Increased default
    model_group.add_argument("--dropout", type=float, default=0.1, help="Model dropout rate")

    # Data args
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1", help="Name of the dataset in `datasets` library (e.g., wikitext-2-raw-v1, wikitext-103-raw-v1)")
    data_group.add_argument("--tokenizer_name", type=str, default="gpt2", help="Name of the tokenizer in `transformers` library")
    data_group.add_argument("--seq_len", type=int, default=256, help="Sequence length for model input")
    data_group.add_argument("--num_proc", type=int, default=os.cpu_count() // 2 or 1, help="Number of processes for dataset mapping (tokenization, filtering). Set to 1 if issues occur.")

    # Training args
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--batch_size", type=int, default=48, help="Training batch size per device") # Moderately increased default
    train_group.add_argument("--train_steps", type=int, default=15000, help="Number of training steps") # Increased default
    train_group.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate") # Standard AdamW LR
    train_group.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    train_group.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1 parameter")
    train_group.add_argument("--adam_beta2", type=float, default=0.98, help="AdamW beta2 parameter (0.98 common for Transformers/Mamba)")
    train_group.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (norm). 0 to disable.")
    train_group.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")


    # Logging, Evaluation & Saving args
    log_eval_save_group = parser.add_argument_group('Logging, Evaluation, Generation & Saving')
    log_eval_save_group.add_argument("--log_interval", type=int, default=50, help="Log training loss every N steps")
    log_eval_save_group.add_argument("--eval_interval", type=int, default=500, help="Evaluate on validation set and generate sample every N steps")
    log_eval_save_group.add_argument("--output_dir", type=str, default="mbds_wikitext_checkpoints", help="Directory to save best and final model checkpoints")
    log_eval_save_group.add_argument("--save_final_model", action='store_true', help="Save the final model state after training completes")

    # Generation args (controlling the model.generate call)
    gen_group = parser.add_argument_group('Generation Settings (during validation)')
    gen_group.add_argument("--generate_sample_prompt", type=str, default="The history of science shows that", help="Prompt for sample generation. Empty string ('') to disable.")
    gen_group.add_argument("--generation_max_new", type=int, default=60, help="Max new tokens for sample generation.")
    gen_group.add_argument("--generation_temp", type=float, default=0.8, help="Temperature for sample generation (controls randomness). Higher is more random.")
    gen_group.add_argument("--generation_ddim_steps", type=int, default=25, help="Number of DDIM steps for faster generation (e.g., 20-50). If 0 or None, uses full DDPM steps from model config.")


    args = parser.parse_args()

    # --- Argument Validation ---
    if args.l_prime <= 0:
        raise ValueError("💥 --l_prime must be positive")
    if args.seq_len <= 0:
        raise ValueError("💥 --seq_len must be positive")
    if args.batch_size <= 0:
        raise ValueError("💥 --batch_size must be positive")
    if args.eval_interval <= 0:
        raise ValueError("💥 --eval_interval must be positive")
    if args.num_proc < 1:
        raise ValueError("💥 --num_proc must be at least 1")
    if args.generation_ddim_steps is not None and args.generation_ddim_steps < 0:
         logging.warning("⚠️ --generation_ddim_steps was negative, setting to None (will use DDPM).")
         args.generation_ddim_steps = None
    elif args.generation_ddim_steps == 0: # Treat 0 as None for DDPM
        args.generation_ddim_steps = None


    # Start the main training process
    main(args)