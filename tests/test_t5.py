
if __name__ == "__main__":
    task = model.create_tasks(texts=["pick up the fork"])
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

    config = AutoConfig.from_pretrained('google-t5/t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base')

    



