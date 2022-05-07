import json
import logging
from os.path import join, exists
import config


def change_fever(is_train=False, sample=None, custom_path=None):
    if is_train:
        filename = join(config.FEVER_SOURCE, "fever_train.jsonl")
    else:
        if custom_path is None:
            filename = join(config.FEVER_SOURCE, "fever_dev.jsonl")
        else:
            filename = join(config.FEVER_SOURCE, custom_path)

    logging.info("Loading fever " + ("train" if is_train else "dev"))

    with open(filename) as f:
        f.readline()
        lines = f.readlines()
    real_id = 1

    with open(join(config.FEVER_SOURCE, "new_fever_dev.jsonl"),'wb') as w_f:
        for line in lines:
            example = json.loads(line)
            example['id'] = real_id
            real_id = real_id + 1
            #(example)
            #print(type(objec))
            w_f.write(json.dumps(example).encode() + '\n'.encode())
    return


change_fever()
