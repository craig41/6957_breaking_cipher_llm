def _llama3_3_70b():
    llama_3_1_model_id = "meta-llama/Llama-3.3-70B-Instruct"
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_3_1_model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    model = transformers.LlamaForCausalLM.from_pretrained(
        llama_3_1_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
        )
    
    config = transformers.LlamaConfig.from_pretrained(llama_3_1_model_id)
    config.rope_scaling = {
        "type": "llama3",
        "factor": 8.0
    }
    def query(messages, n=None):
        import re
        # Single message processing
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
        outputs = model.generate(
            model_inputs, 
            max_new_tokens=32, 
            do_sample=False, 
            temperature=None, 
            top_p=None)
        raw_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('[/INST]')[-1].split('</s>')[0].strip()
        # Clean the response to remove unwanted tokens
        cleaned_response = re.sub(r'<\|eot_id\|>', '', raw_response)
        # Pattern to detect each occurrence of the assistant's response
        pattern = "assistant\n\n"
        # Split the response based on the pattern and grab the last split
        response_splits = re.split(pattern, cleaned_response)
        # The last item should contain the most recent assistant response
        if len(response_splits) > 1:
            last_response = response_splits[-1].split('</s>')[0].strip()
        else:
            last_response = cleaned_response.strip()
        return last_response
    return query