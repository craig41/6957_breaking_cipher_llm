# pip install -q transformers
import os
import time
import torch
import transformers



if __name__ == "__main__":

    start = time.time()
    
    model_id = "CohereForAI/aya-expanse-8b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    
    directory = "data/encoded_limited_lines_aya/"

    for in_file in os.scandir(directory):
        with open(in_file, 'r') as file:
            text = file.read()
            n_grams = text.split('\n')

            prompt_prefix = 'You are a helpful assistant for predicting masked words based on context, Please give the message with your prediction completely replacing the full masked word dont add any explanation or extra text or characters like :Here is the translation; the text is: '
            # prompt_prefix = 'You are a helpful assistant for predicting encoded words based on context, Please give the original text with your prediction replacing the encoded word dont add any explanation or extra text like :Here is the translation: should be like ex:the dog eats its food; the text is: '
            # prompt_prefix = 'This message has an encoded word in it, return the message replacing the encoded word with your translation of the encoded word: '
            # prompt_suffix = '. Respond only with your guess, which should be the same number of words as the message.'


            # n_grams = [prompt_prefix + x for x in n_grams]
            
            
            decoded_outs = []

            for x in n_grams:
                
                chat = []
                
                # chat_line = {"assistant": prompt_prefix, "user": x}
                # chat_line = { 'role': 'user', 'content': prompt_prefix + x + prompt_suffix}
                chat_line = { 'role': 'user', 'content': prompt_prefix + x}
                chat.append(chat_line)
                
                
                # [{'role': 'user', 'content': 'Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz'}]
                
                # print(f"Message: {chat}")

                # Format the message with the chat template
                # messages = [{"role": "user", "content": "Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz"}]
                # print(f"Message: {messages}")
                
                # break
 
                input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
            
                gen_tokens = model.generate(
                    input_ids, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    temperature=0.3,
                    )

                gen_text = tokenizer.decode(gen_tokens[0])
                print(gen_text)
                # Parsing logic to extract chatbot output
                start_token = "<|CHATBOT_TOKEN|>"
                end_token = "<|END_OF_TURN_TOKEN|>"

                start_idx = gen_text.find(start_token) + len(start_token)
                end_idx = gen_text.find(end_token, start_idx)

                clean_text = gen_text[start_idx:end_idx].strip()
                decoded_outs.append(clean_text)
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


        