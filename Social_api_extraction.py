#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install praw


# 1. Reddit Account Login
# 
# First, you need a Reddit account. If you don’t have an account, you’ll need to create one on reddit.com. If you already have an account, log in.
# 
# 2. Go to the App Registration Page
# 
# While logged in, navigate to https://www.reddit.com/prefs/apps. This is Reddit’s "app preferences" page, where you can create an application for API use.
# 
# 3. Create a New Application
# 
# At the bottom of the page, you’ll see a "create app" or "create another app" button. Click this button.
# 
# You’ll need to fill out the following fields:
# 
# - name: Enter a name for your app (e.g., "MyRedditBot").
# - App type: If you’re writing a script for personal use, select "script." For a web app or mobile app, you can choose other options.
# - description: Write a brief description of your app (optional).
# - about url: A URL with more information about your app (optional).
# - redirect uri: For a "script" type app, you can typically use a default value like http://localhost:8080. (This is the address where the API will redirect after a call.)

# In[9]:


import praw
import pandas as pd
import os
from datetime import datetime
import time


date = datetime.today().strftime('%Y-%m-%d')

# Automatically detect Windows user and save with daily basis csv
username = os.getlogin()
EXCEL_FILE_PATH = rf'C:\Users\{username}\Documents\Reddit_Post_{date}.csv



# Add your credentials here
client_id = "get_from_reddit"
client_secret = "get_from_reddit"
user_agent = "get_from_reddit" #e.g. mybot_2026

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Fetch multiple pages of posts
subreddit = reddit.subreddit("select_based_on_needs") #game name? 
posts = []

# Define the number of pages and posts per page
num_pages = 10  # Adjust the number of pages you want
posts_per_page = 100  # Max is 100 per request

# Pagination: Fetch multiple pages using `after`
last_post_id = None  # Keeps track of last post ID

for _ in range(num_pages):
    if last_post_id:
        new_posts = subreddit.new(limit=posts_per_page, params={"after": last_post_id})
    else:
        new_posts = subreddit.new(limit=posts_per_page)
    
    count = 0  # Track number of posts fetched
    for submission in new_posts:
        posts.append({
            "title": submission.title,
            "author": submission.author.name if submission.author else "[deleted]",
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_utc": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'), #deprecated
            #"created_utc": datetime.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "post_url": submission.url,
            "id": submission.id
        })
        last_post_id = submission.fullname  # Correct way to paginate
        count += 1

    print(f"Fetched {count} posts in batch {_ + 1}")

    time.sleep(2)  # Add delay to avoid hitting rate limits

# Convert to DataFrame
df = pd.DataFrame(posts)

# Show results
#df

# Extract file to CSV
def upload_to_sharepoint(file_path):
    """Placeholder function for SharePoint upload"""
    print(f"Uploading {file_path} to folder")

    
# Save to Excel
df.to_csv(EXCEL_FILE_PATH, index=False)
print(f"Data saving completed: {EXCEL_FILE_PATH}")

# Upload to SharePoint (if implemented)
upload_to_sharepoint(EXCEL_FILE_PATH)


# In[ ]:




