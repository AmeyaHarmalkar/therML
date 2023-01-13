import pandas as pd
from tqdm import tqdm
import pickle
import torch
from igfold import IgFoldRunner


#df = pd.read_csv("data/All-TS50-4label.csv", header = 0)
df =  pd.read_csv('data/ts50_updated_isoVH.csv', header = 0)

df_sequences = df["Sequence"]

linkers_list = [
            "GGGGS" * 3,  # G4Sx3
            "GGGGS" * 2,  # G4Sx2
            "GGGGSGGGSGGGGS",  # G4SG3SG4S - likely mistranscription
            "GGGGSGGGGPGGGGS",  # G4SG4PG4S - likely mistranscription
        ]

def split_linkers(seq):
    for linker in linkers_list:
        try:
            heavy, light = seq.split(linker)
        except ValueError:
            continue
        
        return heavy, light
    else:
        raise RuntimeError(f"No appropriate linker found: {seq}")

sequences = []

for seq in df_sequences:
    heavy, light = split_linkers(seq)
    sequences.append( {'H': heavy, 'L': light }  )

#sequences = []
#for i, seq in df.iterrows():
#    sequences.append( {'H': seq['sequence_heavy'], 'L': seq['sequence_light']} )
    
def main():

    igfold = IgFoldRunner()
    embedding = []
    for i in tqdm(range(len(sequences))):
        emb = igfold.embed(
            sequences=sequences[i], # Antibody sequences
        )
        embedding.append(emb.bert_embs.cpu().detach().numpy())
        del emb

    with open('./data/antiberty_emb_isoVH.pkl', 'wb') as f:
        pickle.dump(embedding, f)


if __name__ == '__main__':
    main()