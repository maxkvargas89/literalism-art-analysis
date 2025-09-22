#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 22:53:53 2025

@author: maxvargas
"""

"""

Hypotheses:
    
1.1: Indie films are increasingly reliant on 1:1 art and literalism 
    Source: TheMoviedb    
    Metric: percent of ids
    Criteria: 
        case when movies.budget <= 20000000 then 'indie'
        ** use cpi to convert into 2024 USD
        *** chatGPT agent to analyze movies.overview and assess 1:1 art/literalism
    Test: two-proportion z-test (between decades)
1.2: Blockbusters are increasingly reliant on pre-existing IP
    Metric: percent of total ids
    Criteria: 
        case when movies.budget >= 200000000 then 'blockbuster'
        ** use cpi to convert into 2024 USD
        *** chatGPT agent to analyze movies.overview and assess pre-exisitng IP/sequels
    Test: two-proportion z-test (between decades)
1.3: Blockbusters have a higher return on investment than indie films
    Metric: (movies.revenue - movies.budget) / movies.budget
    Criteria: 
        see above 
    Test: independent samples t-test
1.4: Art galleries are closing at a higher rate than restaraunts (in NY and LA)
    Metric: percent of business_ids that are closed
    Criteria:
        business.categories in ('art gallery')
        business.is_open = 0
        business.city in ("New York City", "Los Angeles")
        review.date >= 2018-01-01
    Test: two-proportion z-test

Starting EDA:
    Time series analysis of Blockbuster budgets and revenue over decades
    Time series analysis of Blockbuster and Indies frequency over decades
    Distribution of ROI
    
"""

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cpi
cpi.update()

# hypothesis 1.1
file = '/Users/maxvargas/literalism_art_analysis/data/random_sampled_us_movies_by_year_enriched.csv'
df = pd.read_csv(file)

## parse dates, extract year, make budget numeric
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year.astype('Int64')
df['decade'] = (df['release_date'].dt.year // 10 * 10)
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df = df[df['year'] < 2025] 

## drop rows with missing years or budgets
df = df.dropna(subset=['year', 'budget'])

## Apply cpi: Get unique years
unique_years = df['year'].dropna().unique()

## Apply cpi: Compute inflation multiplier to 2025 for each year
TARGET_YEAR = 2024
inflation_factors = {
    year: cpi.inflate(1, int(year), to=TARGET_YEAR) for year in unique_years
}

## Apply cpi to budget: Apply inflation multiplier per row
df['adjusted_budget'] = df.apply(
    lambda row: row['budget'] * inflation_factors.get(row['year'], np.nan),
    axis=1
)
df['adjusted_budget'] = df['adjusted_budget'].astype(int)
df['adjusted_budget'] = df['adjusted_budget'].round(2)

## Apply cpi to revenue: Apply inflation multiplier per row
df['adjusted_revenue'] = df.apply(
    lambda row: row['revenue'] * inflation_factors.get(row['year'], np.nan),
    axis=1
)
df['adjusted_revenue'] = df['adjusted_revenue'].astype(int)
df['adjusted_revenue'] = df['adjusted_revenue'].round(2)

## calculate roi
df['roi'] = (df['adjusted_revenue'] - df['adjusted_budget']) * 100 / df['adjusted_budget'] 

## create film_size
df['film_size'] = np.where(df['adjusted_budget'] <= 15000000, 'Indie',
                  np.where(df['adjusted_budget'] > 100000000, 'Blockbuster', 'Other'))

## create two dataframes
df_blockbuster = df[df['film_size'] == 'Blockbuster']
df_indie = df[df['film_size'] == 'Indie']
df_indie = df[df['year'] > 2014]

## create plot dataframes
grouped = df.groupby(['decade','film_size','real_true_stories','adapted_inspired_based']).agg({'id': 'count', 'roi': 'mean'}).reset_index()
grouped_blockbuster = df_blockbuster.groupby(['decade','real_true_stories','adapted_inspired_based']).agg({'id': 'count', 'roi': 'mean'}).reset_index()
grouped_indie = df_indie.groupby(['year','real_true_stories','adapted_inspired_based']).agg({'id': 'count', 'roi': 'mean'}).reset_index()

## create indie real_true_stories decade plot
grouped_indie_pivot = grouped_indie.pivot_table(index='year', columns='real_true_stories', values='id', aggfunc='sum')
indie_year_totals = grouped_indie_pivot.sum(axis=1)
grouped_indie_percent = grouped_indie_pivot.div(indie_year_totals, axis=0) * 100
grouped_indie_percent = grouped_indie_percent.round(2)

## GRAPH Reorder the columns to ensure desired stack order: 'yes' on bottom, then 'no', then 'maybe'
desired_order = ['yes', 'no', 'maybe']
existing_cols = [col for col in desired_order if col in grouped_indie_percent.columns]
grouped_indie_percent = grouped_indie_percent[existing_cols]

ax = grouped_indie_percent.plot.area(figsize=(10, 6), alpha=0.7)

plt.title("Percent of Indies that are Real/True Stories")
plt.xlabel("Year")
plt.ylabel("% of Indies")
plt.xticks(rotation=45)
plt.legend(title="Real or True Story")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.show()

## GRAPH create blockbuster adapted_inspired_based decade plot
grouped_blockbuster_pivot = grouped_blockbuster.pivot_table(index='decade', columns='adapted_inspired_based', values='id', aggfunc='sum')
blockbuster_decade_totals = grouped_blockbuster_pivot.sum(axis=1)
grouped_blockbuster_percent = grouped_blockbuster_pivot.div(blockbuster_decade_totals, axis=0) * 100
grouped_blockbuster_percent = grouped_blockbuster_percent.round(2)

ax = grouped_blockbuster_percent.plot(kind='bar', figsize=(10,6), stacked=True, alpha=0.7)

for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=8, color='white')

plt.title("Percent of Blockbusters that are Adapted/Inspired")
plt.xlabel("Decade")
plt.ylabel("% of Blockbusters")
plt.xticks(rotation=45)
plt.legend(title="Adapted or Inspired Movie")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.show()

## GRAPH create roi heatmap
heatmap_data = grouped_blockbuster.pivot_table(index='decade', columns=['adapted_inspired_based'], values='roi',aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5, linecolor='white')
plt.title("Average ROI by Decade and Adapted/Inspired Movie")
plt.ylabel("Decade")
plt.tight_layout()
plt.show()


