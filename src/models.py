import transformers
import torch


def memoize(f):
    result = [None]
    def memoized_f():
        if result[0] is None:
            result[0] = f()
        return result[0]
    return memoized_f
        
@memoize
def _llama2():
    import transformers
    import torch
    llama_2_model_id = 'meta-llama/Llama-2-70b-chat-hf'
    nf4_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
    model = transformers.LlamaForCausalLM.from_pretrained(
       llama_2_model_id,
        torch_dtype=torch.bfloat16,
        device_map='balanced',
        quantization_config=nf4_config,
        attn_implementation="flash_attention_2",# # # # # use_flash_attention_2=True
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        llama_2_model_id,
        add_bos_token=False)
    
    def query(messages, n=None):
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
        response = tokenizer.batch_decode(model.generate(model_inputs, max_new_tokens=512, do_sample=False, temperature=None, top_p=None, ))[0].split('[/INST]')[-1].split('</s>')[0].strip()
        return response

    return query

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
            max_new_tokens=512, 
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

def _llama3_2_3b():
    llama_3_2_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_3_2_model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    model = transformers.LlamaForCausalLM.from_pretrained(
        llama_3_2_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
        )
    
    config = transformers.LlamaConfig.from_pretrained(llama_3_2_model_id)
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
            max_new_tokens=512, 
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

def _llama3_1_8b_instruct():
    llama_3_1_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
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
            max_new_tokens=512, 
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

def _aya_expanse_8b():
    model_id = "CohereForAI/aya-expanse-8b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

    def query(messages, n=None):
        print(f"Message: {messages}")
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')

        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0
        )
        text = tokenizer.decode(gen_tokens[0])
        return text
    return query


@memoize
def _dummy():
    def query(messages=None, n=None, num_tokens_container:list=None):
        if num_tokens_container is not None:
            num_tokens_container.append((1,1))
        return "No" if n is None else ["No" for _ in range(n)]
    return query

def _gpt2024(openai_api_key=None, helicone_api_key=None, query_kwargs={}):
    from helicone.openai_proxy import openai ; from helicone.globals import helicone_global # # # import openai
    openai.api_key = openai_api_key
    helicone_global.api_key = helicone_api_key

    def query(messages, n=None, query_specific_kwargs={}, num_tokens_container=None):
        if n is None:
            completion = openai.ChatCompletion.create   (
                                                            model="gpt-4-turbo-2024-04-09",
                                                            messages=messages,
                                                            **  {
                                                                    "temperature":0.0,
                                                                    "max_tokens":512,
                                                                    "top_p":1,
                                                                    "frequency_penalty":0,
                                                                    "presence_penalty":0,
                                                                    **query_kwargs,
                                                                    **query_specific_kwargs
                                                                },
                                                        )
            text = completion['choices'][0]['message']['content']
        else:
            completion = openai.ChatCompletion.create   (
                                                            model="gpt-4-turbo-2024-04-09",
                                                            messages=messages,
                                                            n=n,
                                                            seed=74,
                                                            **  {
                                                                    "temperature":1.0,
                                                                    "max_tokens":512,
                                                                    "top_p":1,
                                                                    "frequency_penalty":0,
                                                                    "presence_penalty":0,
                                                                    **query_kwargs,
                                                                    **query_specific_kwargs,
                                                                },

                                                        )
            text = [choice['message']['content'] for choice in completion['choices']]
        if num_tokens_container is not None:
            num_tokens_container.append((completion['usage']['prompt_tokens'], completion['usage']['completion_tokens']))
        return text

    return query

@memoize
def _mistral():
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    def query(messages, n=None):
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()

        generations =   [
                            model.generate  (
                                                model_inputs, 
                                                max_new_tokens=512, 
                                                do_sample=(n is not None), 
                                                temperature=    (
                                                                    0.0 
                                                                    if n is None else 
                                                                    1.0
                                                                ), 
                            )
                            for _ in range  (
                                                1
                                                if n is None else
                                                n
                                            )
                        ]

        responses = [
                        tokenizer.batch_decode(generation)[0].split('[/INST]')[-1].split('</s>')[0].strip()
                        for generation in generations
                    ]
        return responses[0 if n is None else slice(None)]

    return query

@memoize
def _commandrplus():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-plus-4bit")
    model = AutoModelForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-plus-4bit")
    def query(messages, n=None):
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs =  (
                            model_inputs
                            if n is None or n==1 else
                            torch.stack([model_inputs] * n).squeeze()
                        )
        model_inputs = model_inputs.cuda()
        prompt_length = model_inputs.shape[-1]

        generations =   model.generate  (
                                            model_inputs, 
                                            max_new_tokens=512, 
                                            do_sample=(n is not None), 
                                            **(
                                                {
                                                    "temperature":1.0
                                                }
                                                if n is not None else 
                                                {}
                                            )
                        )

        responses = tokenizer.batch_decode(generations[...,prompt_length:], skip_special_tokens=True)
        responses = [response.strip(' ').strip('\n') for response in responses]
        return responses[0 if n is None else slice(None)]
    
    return query

_models =    {
                "mistral"   : _mistral,
                "commandrplus"   : _commandrplus,
                "llama2"     : _llama2,
                "dummy"     : _dummy,
                "gpt2024"   : _gpt2024,
                "llama3_3_70b" : _llama3_3_70b,
                "llama3_2_3b" : _llama3_2_3b,
                "llama3_1_8b_instruct" : _llama3_1_8b_instruct,
                "aya_expanse_8b" : _aya_expanse_8b,
            }   

def load_model(model_name, *args, **kwargs):
    return _models[model_name](*args, **kwargs)