from torch.utils.data import Dataset
import os
import numpy as np
import json



class mel_dataset(Dataset):

    def __init__(self, data_dir):
        
        super(mel_dataset, self).__init__()
        meta_file_path = os.path.join(data_dir, "song_meta.json")
        if os.path.isfile(meta_file_path):
            with open(meta_file_path) as f:
                song_meta = json.load(f)
        else:
            raise FileNotFoundError(f'No such file or directory: {data_dir}/song_meta.json')
        
        song_dict = {}
        genre_dict = {}
        for song in song_meta:
            song_dict[str(song['id'])] = song['song_gn_gnr_basket']
            for i in song['song_gn_gnr_basket']:
                try:
                    genre_dict[i] += 1
                except:
                    genre_dict[i] = 0
                    
        self.genre_count = {k:v for k,v in genre_dict.items()}
        self.genre_index = {k:v for v,k in enumerate(list(genre_dict.keys()))}
        
        result_dict = {}        
        
        for roots, dirs, files in os.walk(data_dir):
            
            listdir = [os.path.join(roots, file) for file in files]
            for i in listdir:
                if ".npy" in i:
                    if np.load(i).shape[1] != 1876:
                        pass
                    else: 
                        try:
                            song_id = i.split('/')[-1].replace('.npy','')
                            result_dict[i] = song_dict[song_id]
                        except:
                            print(song_id,'passed.')
                            
        self.file_list = [key for key in result_dict.keys()]
        label = []

        for genres in result_dict.values():
            one_hot_zero = np.zeros(len(self.genre_index))
            
            for value in genres:
                one_hot_zero[self.genre_index[value]] = 1
                
            label.append(one_hot_zero)

        
        self.label = label
        
    def __getitem__(self, index):
        
        self.x = np.load(self.file_list[index])
        return self.x, self.label[index]
    
    def __len__(self):
        return len(self.file_list)
    
    def genre_index(self):
        return self.genre_index