# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import sqlite3
import numpy as np
import pandas as pd

def demo():
    # reference:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    left = pd.DataFrame({'person_id': ['ID1', 'ID2', 'ID3', 'ID4'], 'product_id': ['P1', 'P1', 'P2', 'P2']}).set_index('person_id')
    right = pd.DataFrame({'person_id': ['ID1', 'ID2', 'ID3', 'ID5'], 'store_id': ['S1', 'S2', 'S3', 'S4']}).set_index('person_id')
    left_join = left.join(right, on=['person_id'], how='left')
    print(f"left join: {left_join}")
    inner_join = left.join(right, on=['person_id'], how='inner')

        # in-memory SQLite database
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
        # https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91

    conn = sqlite3.connect(":memory:")
    left.to_sql('left', con=conn)
    right.to_sql('right', con=conn)

    c = conn.cursor()
    c.execute("SELECT * FROM left LEFT JOIN right ON left.person_id=right.person_id")
    print(f"left join: {c.fetchall()}")

    # save changes
    conn.commit()
    # close the connection if done.
    conn.close()

