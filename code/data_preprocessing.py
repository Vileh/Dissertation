from wildlife_datasets import datasets
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.image as mpimg
from PIL import Image

def download_heads_dataset():

    datasets.SeaTurtleIDHeads.get_data('../data/SeaTurtleIDHeads')

    ds = datasets.SeaTurtleIDHeads('../data/SeaTurtleIDHeads')

    # Convert date to datetime
    ds.df['date'] = pd.to_datetime(ds.df['date'])

    return ds.df

def create_splits_dataframe():
    df = download_heads_dataset()

    # Create time-aware split
    def get_time_aware_split(row):
        if row['date'] < datetime.datetime(2019, 1, 1):
            return 'train'
        elif row['date'] >= datetime.datetime(2020, 1, 1) and row['identity'] in individuals_in_train:
            return 'test'
        elif row['date'] >= datetime.datetime(2019, 1, 1) and row['date'] < datetime.datetime(2020, 1, 1) and row['identity'] in individuals_in_train:
            return 'val'
        else:
            return 'ignore'

    def get_encounter_split(group):
        total_images = len(group)
        encounters_df = group.groupby('date').size().sort_index()
        
        if len(encounters_df) <= 3:
            split_labels = ['train', 'test', 'val']
            for i, encounter in enumerate(encounters_df.index):
                group.loc[group['date'] == encounter, 'time-proportional'] = split_labels[i]
        else:
            cumulative_images = encounters_df.cumsum()
            train_threshold = int(total_images * 0.75)
            val_threshold = int(total_images * 0.85)
            
            train_end_date = cumulative_images[cumulative_images > train_threshold].index[0]
            val_end_date = cumulative_images[(cumulative_images > train_threshold) & (cumulative_images > val_threshold)].index[0] if any((cumulative_images > train_threshold) & (cumulative_images > val_threshold)) else train_end_date
            
            def assign_split(date):
                if date <= train_end_date:
                    return 'train'
                elif date <= val_end_date:
                    return 'val'
                else:
                    return 'test'
            
            group['time-proportional'] = group['date'].apply(assign_split)
        
        return group
    
    def random_split(group):
        split_labels = ['train', 'val', 'test']
        split_probabilities = [0.75, 0.10, 0.15]

        # Ensure at least one image per individual is in the training set
        if len(group) == 1:
            group['random'] = 'train'
        else:
            group.loc[group.sample(n=1).index, 'random'] = 'train'

            # Randomly assign the remaining images to the splits
            remaining_indices = group[group['random'].isna()].index
            remaining_labels = np.random.choice(split_labels, size=len(remaining_indices), p=split_probabilities)
            group.loc[remaining_indices, 'random'] = remaining_labels

        return group

    df['time-aware-train'] = df['date'] < datetime.datetime(2019, 1, 1)
    individuals_in_train = df[df['time-aware-train']]['identity'].unique()
    df['time-aware'] = df.apply(get_time_aware_split, axis=1)
    df = df.groupby('identity', group_keys=False).apply(get_encounter_split)
    df = df.groupby('identity', group_keys=False).apply(random_split)

    df.drop(columns=['time-aware-train'], inplace=True)

    return df

def get_data_label_map(df):
    label_map = {'time-aware': {identity: i for i, identity in enumerate(df[df['time-aware'] == 'train']['identity'].unique())},
             'time-proportional': {identity: i for i, identity in enumerate(df[df['time-proportional'] == 'train']['identity'].unique())},
             'random': {identity: i for i, identity in enumerate(df[df['random'] == 'train']['identity'].unique())}}

    inversed_label_map = {split: {i: identity for identity, i in label_map[split].items()} for split in label_map}

    return label_map, inversed_label_map

def split_into_folders(df, folder):
    folder_path = f"../data/SeaTurtleIDHeads/{folder}"

    if not os.path.exists(folder_path):
        for index in range(len(df)):
            turtle = df.iloc[index]

            # Load the image
            img = mpimg.imread(f"../data/SeaTurtleIDHeads/{turtle['path']}")

            # Convert the cropped image to a PIL Image
            cropped_img_pil = Image.fromarray((img).astype(np.uint8))

            # Resize the image to 224x224
            rescaled_img = cropped_img_pil.resize((224, 224))

            for split in ['time-aware', 'time-proportional', 'random']:
                curr_split = turtle[split]

                identity = turtle['identity']

                # Create the directory if it doesn't exist
                path = f"{folder_path}/{split}/{curr_split}/{identity}"
                os.makedirs(path, exist_ok=True)

                # Save image
                rescaled_img.save(f"{path}/{index}.jpg")

def print_split_summary(df, split_column):
    print(f"Summary for {split_column}:")

    # Number of images per split
    split_counts = df[split_column].value_counts()
    print("Number of images per split:")
    print(split_counts)

    # Proportions of train/val/test splits
    total_images = len(df)
    split_proportions = split_counts / total_images
    print("Proportions of train/val/test splits:")
    print(split_proportions)

    # Number of individuals in train split
    train_individuals = df[df[split_column] == 'train']['identity'].nunique()
    print(f"Number of individuals in train split: {train_individuals}")

    # Number of individuals in val split
    val_individuals = df[df[split_column] == 'val']['identity'].nunique()
    print(f"Number of individuals in val split: {val_individuals}")

    # Number of individuals in test split
    test_individuals = df[df[split_column] == 'test']['identity'].nunique()
    print(f"Number of individuals in test split: {test_individuals}")

    print()