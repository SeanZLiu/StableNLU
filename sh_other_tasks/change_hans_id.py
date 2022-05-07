from os.path import join
import config

def change_hans() :
    src = join(config.HANS_SOURCE, "former_heuristics_evaluation_set.txt")
    target = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")

    with open(src, "r") as f, open(target, "wb") as writo_f:
        first_line = f.readline()
        writo_f.write(first_line.encode())
        lines = f.readlines()
        id = 1
        for line in lines:
            parts = line.split("\t")
            parts[7] = str(id)
            id = id + 1
            writo_f.write(("\t".join(parts)).encode())
    return

change_hans() 
