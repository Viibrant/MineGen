"""Generates a csv containing all block ids"""

import minecraft_data
import pandas as pd

if __name__ == "__main__":
    mcd = minecraft_data("1.19")
    json = mcd.blocks_list
    df = pd.DataFrame(json)
    df.to_csv("block_ids.csv", index=False)
