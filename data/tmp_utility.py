

def prepare_for_phonetic_transcription(txt_file: str, w_t_s: dict):
    with open(txt_file, 'w') as f:
        for key, item in w_t_s.items():
            for i in item: 
                f.write(i)


