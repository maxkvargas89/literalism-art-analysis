#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 22:53:53 2025

@author: maxvargas
"""

"""
Hypothesis 1.1: Indie films are less frequently created today
    Metric: count of ids
    Criteria: 
        if budget <= $2M then "indie" 
        elseif budget >= $100M then "blockbuster" 
        else "other"
Hypothesis 1.2: Blockbusters have an increasingly higher share of the box office 
    Metric: percent of total ids
    Criteria: 
        if budget <= $2M then "indie" 
        elseif budget >= $100M then "blockbuster" 
        else "other"
Hypothesis 1.3: Blockbusters have a higher return on investment than indie films
    Metric: ROI (revenue - budget) / budget
    Criteria: 
        if budget <= $2M then "indie" 
        elseif budget >= $100M then "blockbuster" 
        else "other"
Hypothesis 1.4: Art galleries are closing at a higher rate than restaraunts (in NY and LA)
    Metric: percent of businesses that are closed
    Criteria:
        business category is art gallery
        business opened >= 2018
        business location in ("NY", "LA")
"""

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cpi
cpi.update()

# hypothesis 1.1
file = '/Users/maxvargas/literalism_art_analysis/data/random_sampled_movies_by_year.csv'
df = pd.read_csv(file)

## parse dates, extract year, make budget numeric
df['release_date'] = pd.to_datetime(df['release_date'])
df['year'] = df['release_date'].dt.year.astype('Int64')
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

## drop rows with missing years or budgets
df = df.dropna(subset=['year', 'budget'])

## Adjust budgets for inflation
def adjust_budget(row):
    if pd.notna(row['budget']) and pd.notna(row['year']):
        try:
            return cpi.inflate(row['budget'], int(row['year']))
        except:
            return np.nan
    return np.nan

df['adjusted_budget'] = df.apply(adjust_budget, axis=1)

## create film_size
df['film_size'] = np.where(df['adjusted_budget'] <= 20000000, 'Indie',
                  np.where(df['adjusted_budget'] >= 200000000, 'Blockbuster', 'Other'))

## create plot dataframe
df_budget = df.groupby(['film_size','year'])['id'].count().reset_index()
df_budget = df_budget[df_budget['film_size'] == 'Indie']

## create plot
df_pivot = df_budget.pivot(index='year', columns='film_size', values='id')
df_pivot.plot(kind='line', marker='o', figsize=(10,6))
plt.title("Number of Indie Movies by Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.grid(True)
plt.tight_layout()
plt.show()
