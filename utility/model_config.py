from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import json

def get_model_config(model_name):
    """
    Get the configuration of a Hugging Face model
    """
    print(f"Loading configuration for: {model_name}")
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Print key configuration details
    print("\n" + "="*50)
    print("MODEL CONFIGURATION")
    print("="*50)
    
    print(f"Model Type: {config.model_type}")
    print(f"Architecture: {config.architectures}")
    print(f"Vocabulary Size: {config.vocab_size}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Number of Layers: {config.num_hidden_layers}")
    print(f"Number of Attention Heads: {config.num_attention_heads}")
    print(f"Intermediate Size: {config.intermediate_size}")
    print(f"Maximum Position Embeddings: {config.max_position_embeddings}")
    print(f"RMS Norm Epsilon: {getattr(config, 'rms_norm_eps', 'N/A')}")
    print(f"Rope Theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"Attention Dropout: {getattr(config, 'attention_dropout', 'N/A')}")
    print(f"Hidden Dropout: {getattr(config, 'hidden_dropout_prob', 'N/A')}")
    
    # Print full config as JSON
    print("\n" + "="*50)
    print("FULL CONFIGURATION (JSON)")
    print("="*50)
    print(json.dumps(config.to_dict(), indent=2))
    
    return config

def get_tokenizer_info(model_name):
    """
    Get tokenizer information
    """
    print(f"\n{'='*50}")
    print("TOKENIZER INFORMATION")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Tokenizer Type: {type(tokenizer).__name__}")
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Model Max Length: {tokenizer.model_max_length}")
    print(f"Padding Token: {tokenizer.pad_token}")
    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"BOS Token: {tokenizer.bos_token}")
    print(f"UNK Token: {tokenizer.unk_token}")
    
    return tokenizer

def get_model_info(model_name):
    """
    Get basic model information without loading the full model
    """
    print(f"\n{'='*50}")
    print("MODEL INFORMATION")
    print("="*50)
    
    # This will show model info without downloading the full model weights
    try:
        # Just get config and tokenizer info
        config = get_model_config(model_name)
        tokenizer = get_tokenizer_info(model_name)
        
        # Calculate approximate parameter count
        if hasattr(config, 'num_hidden_layers') and hasattr(config, 'hidden_size'):
            # Rough estimation for transformer models
            vocab_params = config.vocab_size * config.hidden_size
            layer_params = config.num_hidden_layers * (
                4 * config.hidden_size * config.hidden_size +  # attention weights
                config.hidden_size * config.intermediate_size * 2  # feed forward
            )
            total_params = vocab_params + layer_params
            print(f"\nEstimated Parameters: ~{total_params:,} ({total_params/1e9:.2f}B)")
        
        return config, tokenizer
        
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None, None

if __name__ == "__main__":
    model_name = "mistralai/Mathstral-7b-v0.1"
    
    print("Getting model configuration and information...")
    config, tokenizer = get_model_info(model_name)