import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config, latest_weights_file_path

from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

import warnings
from tqdm import tqdm
from tabulate import tabulate 
from pathlib import Path

TOKEN = ""

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])

        # Select the token with the max probability (Greedy Search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=5):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output. This is done only once and reused for each decoding step.
    encoder_output = model.encode(source, source_mask)

    # Initialize the beams.
    # A beam is a hypothesis, which is a tuple of (sequence_tensor, log_probability_score).
    # We start with a single beam: the SOS token with a log probability of 0 (since log(1) = 0).
    beams = [(torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device), 0.0)]
    
    # This list will store hypotheses that have finished (i.e., ended with an EOS token).
    completed_hypotheses = []

    # Start the decoding loop, which runs for a maximum of `max_len` steps.
    for _ in range(max_len):
        new_beams = []
        
        # Iterate through each current hypothesis (beam) to expand it.
        for hypothesis, score in beams:
            
            # Check if the last token of the hypothesis is the EOS token.
            if hypothesis[0, -1].item() == eos_idx:
                # If it is, this hypothesis is complete. Move it to the completed list
                # and do not expand it further.
                completed_hypotheses.append((hypothesis, score))
                continue

            # --- Expand the current hypothesis ---
            
            # Prepare the decoder input and mask for the current hypothesis.
            decoder_input = hypothesis
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            
            # Get the log probabilities for the next token from the model.
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            log_probs = model.project(out[:, -1]) # The model's projection layer already uses log_softmax.
            
            # Get the top `beam_size` most likely next tokens and their log probabilities.
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)

            # Create `beam_size` new hypotheses by appending each of the top-k tokens.
            for i in range(beam_size):
                next_word_idx = topk_indices[0, i].item()
                log_prob = topk_log_probs[0, i].item()

                # Create the new sequence tensor.
                new_hypothesis = torch.cat(
                    [hypothesis, torch.empty(1, 1).type_as(source).fill_(next_word_idx).to(device)], dim=1
                )
                
                # Add the new hypothesis to the list of candidates.
                # The new score is the sum of the old score and the log probability of the new token.
                new_beams.append((new_hypothesis, score + log_prob))

        # If there are no new beams to consider (e.g., all hypotheses ended in EOS), stop.
        if not new_beams:
            break
            
        # Prune the beams.
        # From all the newly generated candidate hypotheses, select the top `beam_size` best ones.
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
    # After the loop, add any unfinished beams from the final step to the completed list.
    # This handles cases where the generation stops due to reaching `max_len`.
    completed_hypotheses.extend(beams)

    # If no hypotheses were ever completed, return the best one (if have)
    if not completed_hypotheses:
        if beams:
            best_hypothesis = max(beams, key=lambda x: x[1] / len(x[0].squeeze(0)))
        else: 
            return torch.tensor([sos_idx, eos_idx]).long().to(device)
    else:
        best_hypothesis = max(completed_hypotheses, key=lambda x: x[1] / x[0].size(1))

    # Make sure the output is never just SOS
    output_tokens = best_hypothesis[0].squeeze(0)

    if len(output_tokens) <= 1:
        return torch.tensor([sos_idx, eos_idx]).long().to(device)

    return output_tokens


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, 
                   device, print_msg, global_step, writer, num_examples=5):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Size of the control window
    console_width = 80

    with torch.no_grad():
        for i, batch in enumerate(validation_ds):
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            model_out = beam_search_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=5)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            cleaned_prediction = model_out_text.replace(" ##", "").replace("##", "")

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(cleaned_prediction)

            # Print to the console
            if print_msg:
                print_msg('-'*console_width)
                print_msg(f"({i+1}/{num_examples})")
                print_msg(f"SOURCE: {source_text}")
                print_msg(f"TARGET: {target_text}")
                print_msg(f"PREDICTED (RAW): {model_out_text}")
                print_msg(f"PREDICTED (CLEAN): {cleaned_prediction}")

            if count == num_examples:
                break

    metric_cer = CharErrorRate()
    cer = metric_cer(predicted, expected)
    metric_wer = WordErrorRate()
    wer = metric_wer(predicted, expected)
    metric_bleu = BLEUScore()
    bleu = metric_bleu(predicted, expected)

    if print_msg:
        print_msg("\n" + '='*console_width)
        print_msg(f"EVALUATION METRICS ON {num_examples} EXAMPLES")
        metrics_data = [
            ["Character Error Rate (CER)", f"{cer.item():.4f}"],
            ["Word Error Rate (WER)", f"{wer.item():.4f}"],
            ["BLEU Score", f"{bleu.item():.4f}"]
        ]
        print_msg(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))
        print_msg('='*console_width)

    if writer:
        writer.add_scalar('validation/cer', cer, global_step)
        writer.add_scalar('validation/wer', wer, global_step)
        writer.add_scalar('validation/bleu', bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        print(f"Building WordPiece tokenizer for {lang}...")
        tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
        trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=64000)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw_train = load_dataset('opus100', 'en-ja', split = 'train', token=TOKEN)
    ds_raw_val = load_dataset('opus100', 'en-ja', split = 'validation', token=TOKEN)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw_train, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw_train, config['lang_tgt'])


    train_ds = BilingualDataset(ds_raw_train, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(ds_raw_val, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0

    # Combine 2 sets to examine
    combined_ds = concatenate_datasets([ds_raw_train, ds_raw_val])
    for item in combined_ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True,  num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],
                              config['d_model'])
    return model


def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    if torch.__version__.startswith("2."):
        print("Compiling the model with torch.compile...")
        model = torch.compile(model)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    scaler = GradScaler()

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        total_loss = 0.0
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (b, 1, seq_len, seq_len)

            use_amp = device.type == 'cuda'
            with autocast(device_type=device.type, enabled=use_amp):
                # Run the tensors through the transformer
                encoder_output = model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (b, seq_len, d_model)
                proj_output = model.project(decoder_output) # (b, seq_len, tgt_vocab_size)

                label = batch['label'].to(device) # (b, seq_len)

                # (b, seq_len, tgt_vocab_size) --> (b * seq_len, tgt_vocab_size)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1).long())
            
            total_loss += loss.item()
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            scaler.scale(loss).backward()

            # Update the weights
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            global_step += 1 
        
        avg_epoch_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch:02d} - Average Training Loss: {avg_epoch_loss:.4f}")

        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], 
                       device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
