#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:58:17 2025

@author: maxvargas
"""

import requests
import pandas as pd
import time
import random
import os
from datetime import datetime, timedelta

# Config
API_KEY = "cebeceb26a762ee305f9c4c6384ea3d6"
BASE_URL = "https://api.themoviedb.org/3"
YEARS = list(range(2000, 2025, 1))  # Spread across decades
DATES_PER_YEAR = 3                 # Number of random release windows per year
DAYS_RANGE = 60                    # Length of each sampling window
MOVIES_PER_SAMPLE = 20            # Max movies per page
SLEEP = 0.4                        # Sleep between requests
SORT_OPTIONS = [
    "popularity.desc", "vote_average.desc", "revenue.desc",
    "release_date.desc", "vote_count.desc"
]

# HELPER FUNCTIONS
def get_random_window(year):
    """Generate a random 30-day window within a given year."""
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 1)
    rand_day = random.randint(0, (end - start).days)
    window_start = start + timedelta(days=rand_day)
    window_end = window_start + timedelta(days=DAYS_RANGE)
    return window_start.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')

def get_genre_names(genres):
    return ", ".join([g["name"] for g in genres])

def get_country_names(production_companies):
    return ", ".join([c["origin_country"] for c in production_companies])

def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        d = r.json()
        return {
            "id": d.get("id"),
            "title": d.get("title"),
            "release_date": d.get("release_date"),
            "overview": d.get("overview"),
            "tagline": d.get("tagline"),
            "runtime": d.get("runtime"),
            "genres": get_genre_names(d.get("genres", [])),
            "origin_country": get_country_names(d.get("production_companies", [])),
            "rating": d.get("vote_average"),
            "revenue": d.get("revenue"),
            "budget": d.get("budget"),
            "popularity": d.get("popularity"),
            "vote_average": d.get("vote_average"),
            "vote_count": d.get("vote_count")
        }
    return None

# MAIN SCRIPT
movie_data = []

print(f"🎬 Randomly sampling movies across {len(YEARS)} years...")

for year in YEARS:
    print(f"\n📅 Year: {year}")
    for i in range(DATES_PER_YEAR):
        gte, lte = get_random_window(year)
        sort_by = random.choice(SORT_OPTIONS)
        page = random.randint(1, 5)  # Use lower range to avoid missing data in older years

        print(f"🔄 Fetching: {gte} to {lte} | sort={sort_by} | page={page}")
        discover_url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "primary_release_date.gte": gte,
            "primary_release_date.lte": lte,
            "sort_by": sort_by,
            "page": page,
            "language": "en-US"
        }

        r = requests.get(discover_url, params=params)
        if r.status_code == 200:
            results = r.json().get("results", [])
            random.shuffle(results)  # Extra shuffle within result set
            for m in results:
                details = get_movie_details(m["id"])
                if details:
                    movie_data.append(details)
                    print(f"✅ {details['title']} ({details['release_date']})")
                time.sleep(SLEEP)
        else:
            print(f"❌ Failed discover fetch: {r.status_code}")
        time.sleep(SLEEP)
  
# SAVE RESULTS
df = pd.DataFrame(movie_data)
OUTPUT_PATH = "/Users/maxvargas/code_testing/data/random_sampled_movies_by_year_v2.csv"
if os.path.exists(OUTPUT_PATH):
    df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
else:
    df.to_csv(OUTPUT_PATH, mode='w', header=True, index=False)
