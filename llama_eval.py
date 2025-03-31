import os
import re
import time
import transformers
import torch
from huggingface_hub import login
    

from constants import Constants

model_id = "meta-llama/Llama-3.1-70B-Instruct"

access_token = Constants.HUGGINGFACE_TOKEN_2



if __name__ == "__main__":
    
    start = time.time()
    login(token=access_token)
    
    llama_3_1_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_3_1_model_id, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token 
    model = transformers.LlamaForCausalLM.from_pretrained(
        llama_3_1_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=access_token 
        )
    
    config = transformers.LlamaConfig.from_pretrained(llama_3_1_model_id)
    config.rope_scaling = {
        "type": "llama3",
        "factor": 8.0
    }
    
    directory = "data/encoded_limited_lines/"

    for in_file in os.scandir(directory):
        with open(in_file, 'r') as file:
            text = file.read()
        n_grams = text.split('\n')
        
        # prompt_prefix = "This message has an encoded word in it, can you tell me what the encoded word is by responding with the entire message replacing the encoded word with the actual word, for example if I give you the message 'my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith' and the encoded word is 'name', then return 'my name is John Smith': "
        
        prompt_prefix = "You are a robot that only responds with strings of similar length to the question. Example question: my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith. Example Answer: my name is John Smith. Now here is my question: "

        n_grams = [prompt_prefix + x for x in n_grams]
        
        inputs = []
        decoded_outs = []
        
        chat = []
        
        # "You are a robot that only responds with strings of similar length to the question. Example question: my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith. Example Answer: my name is John Smith. Now here is my question: " 
        
        # chat.append({"role": "system", "content": "This message has an encoded word in it, can you tell me what the encoded word is by responding with the entire message replacing the encoded word with the actual word, for example if I give you the message my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith and the encoded word is name, then return my name is John Smith"})

        l = 0
        for x in n_grams:
            l += 1
            chat_line = {"role": "user", "content": x}
            chat.append(chat_line)
            # print(chat_line)
            # if l == 2:
            # break
        
        # chat = [
        #     {"role": "user", "content": "Hello, how are you?"},
        #     {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        #     {"role": "user", "content": "I'd like to show off how chat templating works!"},
        # ]
            
            tokenized_chat = tokenizer.apply_chat_template(chat, return_tensors="pt").to('cuda')
        
        # outputs = model.generate(tokenized_chat)

            # # Single message processing
            # model_inputs = tokenizer.apply_chat_template(x, return_tensors="pt").to('cuda')
            outputs = model.generate(
                tokenized_chat, 
                max_new_tokens=32, 
                do_sample=False, 
                temperature=None, 
                top_p=None)
            
            # for o in outputs:
            raw_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print("raw_response: " + raw_response)
            raw_response = raw_response.split('[/INST]')[-1].split('</s>')[0].strip()
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
                
            decoded_outs.append(last_response)
            
        end = time.time()

        print("Time taken: ", end-start)

        orig_filename = os.path.basename(in_file)
        orig_filename = os.path.splitext(orig_filename)[0]
        filename = "data/llama_pred_partial/" + orig_filename + ".txt"

        with open(filename, "w") as txt_file:
            for line in decoded_outs:
                txt_file.write(line + "\n")