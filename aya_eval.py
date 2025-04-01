# pip install -q transformers
import os
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM



if __name__ == "__main__":

    start = time.time()
    
    model_id = "CohereForAI/aya-expanse-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    directory = "data/encoded_limited_lines/"

    for in_file in os.scandir(directory):
        with open(in_file, 'r') as file:
            text = file.read()
            n_grams = text.split('\n')

            prompt_prefix = "This message has an encoded word in it, can you tell me what the encoded word is by responding with the entire message replacing the encoded word with the actual word, for example if I give you the message 'my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith' and the encoded word is 'name', then return 'my name is John Smith': "

            n_grams = [prompt_prefix + x for x in n_grams]
            
            chat = []
            decoded_outs = []
            
            for x in n_grams:
                
                chat_line = {"role": "user", "content": x}
                chat.append(chat_line)
                

                # Format the message with the chat template
                messages = [{"role": "user", "content": "Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz"}]
                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                ## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

                gen_tokens = model.generate(
                    input_ids, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=0.3,
                    )

                gen_text = tokenizer.decode(gen_tokens[0])
                decoded_outs.append(gen_text)
                # print(gen_text)
                
            end = time.time()

            print("Time taken: ", end-start)

            orig_filename = os.path.basename(in_file)
            orig_filename = os.path.splitext(orig_filename)[0]
            filename = "data/aya_pred_partial/" + orig_filename + ".txt"

            with open(filename, "w") as txt_file:
                for line in decoded_outs:
                    txt_file.write(line + "\n")
    
    #----------
    

    

    # for in_file in os.scandir(directory):
    #     with open(in_file, 'r') as file:
    #         text = file.read()
    #     n_grams = text.split('\n')

    #     prompt_prefix = "This message has an encoded word in it, can you tell me what the encoded word is by responding with the entire message replacing the encoded word with the actual word, for example if I give you the message 'my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith' and the encoded word is 'name', then return 'my name is John Smith': "

    #     n_grams = [prompt_prefix + x for x in n_grams]

    #     inputs = []
    #     decoded_outs = []

    #     for x in n_grams:
    #         inputs.append(tokenizer.encode(x, return_tensors="pt"))


    #     for i in inputs:
    #         with torch.no_grad():
    #             output = aya_model.generate(i, max_new_tokens=128)
    #             decoded_out = tokenizer.decode(output[0])
    #         decoded_outs.append(decoded_out)


        