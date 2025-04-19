from tqdm import tqdm
import torch
import os


from .fn_utils import (
    causal_mask, generate_unique_random_integers, 
    get_model, decode_sequence)

from modeling.constants import BOS_IDX, PAD_IDX, EOS_IDX

class Predictor:
    """
    Class for generating predictions using a trained model.

    Args:
        config (object): Configuration object containing model and inference settings.
        load_best (bool, optional): Whether to load the best model. Defaults to True.
        epoch (int, optional): Epoch number to load a specific checkpoint.

    Attributes:
        model (Model): Trained model for prediction.
        path (str): Path to the trained model.
        device (str): Device for inference.
        checkpoint (str): Model checkpoint path.
        max_len (int): Maximum target sequence length for inference.
    """

    def __init__(self, config, load_best=True, epoch=None):
        self.model = get_model(config)
        
        # Determine checkpoint path
        if load_best:
            self.checkpoint = f"{config.model_name}_best.pth"
        else:
            self.checkpoint = f"{config.model_name}_ep{epoch + 1}.pth"
        
        self.path = os.path.join(config.root_dir, self.checkpoint)
        
        # Set device for inference
        self.device = (
            f"cuda:{config.device}" if "cuda" not in str(config.device) else config.device
        )
        
        # Load model state
        state = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
        
        # Maximum target length for inference
        self.max_len = config.tgt_max_len
        
        print(f"Using epoch {state['epoch']} model for predictions.")

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        """
        Generate a sequence using greedy decoding.

        Args:
            src (Tensor): Source input.
            src_mask (Tensor): Mask for source input.
            max_len (int): Maximum length of the generated sequence.
            start_symbol (int): Start symbol for decoding.

        Returns:
            Tensor: Generated sequence.
        """
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        src = src.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)
        memory = self.model.encode(src, src_mask)
        memory = memory.to(self.device)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for _ in range(max_len - 1):
            tgt_mask =(causal_mask(ys.size(1)).type(torch.bool)).to(self.device)
            tgt_mask = tgt_mask.unsqueeze(0)
            out = self.model.decode(memory,src_mask,ys,tgt_mask)
            prob = self.model.project(out[:,-1])

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == EOS_IDX:
                break
        return ys

    def predict(self, test_example, itos, raw_tokens=False):
        """
        Generate prediction for a test example.

        Args:
            test_example (dict): Test example containing input features.
            raw_tokens (bool, optional): Whether to return raw tokens. Defaults to False.

        Returns:
            str or tuple: Decoded equation or tuple of original and predicted tokens.
        """
        self.model.eval()

        src = test_example[0]

        src_mask = (src != PAD_IDX).unsqueeze(0).unsqueeze(0)
        tgt_tokens = self.greedy_decode(
            src, src_mask, max_len=self.max_len, start_symbol=BOS_IDX).flatten()

        if raw_tokens:
            original_tokens = test_example[1]
            return original_tokens, tgt_tokens

        decoded_eqn = ''
        for t in tgt_tokens:
            decoded_eqn += itos[int(t)]

        return decoded_eqn


def sequence_accuracy(config,test_ds,tgt_itos,load_best=True, epoch=None,test_size=100):
    """
    Calculate the sequence accuracy.

    Args:
        load_best (bool, optional): Whether to load the best model. Defaults to True.
        epochs (int, optional): Number of epochs. Defaults to None.

    Returns:
        float: Sequence accuracy.
    """
    predictor = Predictor(config,load_best, epoch)
    count = 0
    num_samples = 10 if config.debug else test_size 
    random_idx = generate_unique_random_integers(
        num_samples, start=0, end=len(test_ds))
    length = len(random_idx)
    pbar = tqdm(range(length))
    pbar.set_description("Seq_Acc_Cal")
    for i in pbar:
        original_tokens, predicted_tokens = predictor.predict(
            test_ds[random_idx[i]],tgt_itos, raw_tokens=True)
        original_tokens = original_tokens.detach().numpy().tolist()
        predicted_tokens = predicted_tokens.detach().cpu().numpy().tolist()
        original = decode_sequence(original_tokens,tgt_itos)
        predicted = decode_sequence(predicted_tokens,tgt_itos)
        if original == predicted:
            count = count + 1
        pbar.set_postfix(seq_accuracy=count / (i + 1))
    return count / length
