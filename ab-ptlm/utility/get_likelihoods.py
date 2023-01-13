import pandas as pd
from tqdm import tqdm
import pickle
import ablang


df = pd.read_csv("data/all_ts50_data.csv", header = 0)

label_dict = {'CCR8' : 'SetA',
             'CD123' : 'SetB',
             'CD20': 'SetQ',
             'CD22starting' : 'SetC',
             'CD22AFFMAT' : 'SetC',
             'CD70' : 'SetF',
             'CLDN18.2' : 'SetD',
             'CLL1': 'SetK',
             'CLL1opt' : 'SetK',
             'CS16C12Opt' : 'SetH',
             'CS115G8Opt' : 'SetH',
             'CS1' : 'SetH',
             'DCAF4L2' : 'SetL',
             'EpCAM' : 'SetM',
             'LRRC15' : 'SetG',
             'MAGEB2': 'SetI',
             'MUC1-SEA' : 'SetE',
             'MUC13' : 'SetP',
             'MUC16' : 'SetN',
             'MUC17' : 'SetO',
             'PSMAAFFMAT' : 'SetJ',
             'test_I2C' : 'Test scFv',
             'test-ISO-VH' : 'Isolated scFv'}


heavy_ablang = ablang.pretrained("heavy")
heavy_ablang.freeze()
light_ablang = ablang.pretrained("light")
light_ablang.freeze()



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


likelihoods = {}

for target in label_dict.keys():

    print(target)
    temp_df = df[ df["Group"] == target ]

    data_list = []

    for i, seq in temp_df.iterrows():
        heavy, light  =  split_linkers(seq["Sequence"])
        p = {}
        p['H'] = heavy_ablang( heavy, mode='likelihood' )
        p['L'] = light_ablang( light, mode='likelihood' )
        data_list.append( [ seq["Group"], seq["Name"], seq["label"], seq["TS50_float"], p ] )

    likelihoods[target] = data_list


with open('./data/updated_ablang_likelihoods.pkl', 'wb') as f:
        pickle.dump(likelihoods, f)