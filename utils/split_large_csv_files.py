def to_csv_batch(src_csv, dst_dir, size=90000, index=False):

    import math

    import pandas as pd

    # Read source csv
    df = pd.read_csv(src_csv)

    # Initial values
    low = 0
    high = size

    # Loop through batches
    for i in range(math.ceil(len(df) / size)):

        fname = dst_dir + "/Batch_" + str(i + 1) + ".csv"
        df[low:high].to_csv(fname, index=index)

        # Update selection
        low = high
        if high + size < len(df):
            high = high + size
        else:
            high = len(df)


# split(open('data/WDI_CSV/WDICSV.csv', 'r'))
to_csv_batch("data/WDI_CSV/WDICSV.csv", "data/Batches")
