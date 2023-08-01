
#household files 
from base import df, interface

#Top Section------------------------------

#Likes
# Checking if the 'LikesCount' column exists in the DataFrame
if 'LikeCount' in df.columns:
    # Calculating the sum of likes
    total_likes = df['LikeCount'].sum()
    # Setting the text to display the total likes
    likes_text = str(total_likes)
else:
    # Setting the text to display 'Data Unavailable'
    likes_text = 'Data Unavailable'

#Retweets
# Checking if the 'RetweetCount' column exists in the DataFrame
if 'RetweetCount' in df.columns:
    # Calculating the sum of retweets
    total_retweet = df['RetweetCount'].sum()
    # Setting the text to display the total retweet
    retweet_text = str(total_retweet)
else:
    # Setting the text to display 'Data Unavailable'
    retweet_text = 'Data Unavailable'

#Reshare block
# Checking if the 'RetweetCount' column exists in the DataFrame
if 'ReshareCount' in df.columns:
    # Calculating the sum of shares
    total_reshare = df['ShareCount'].sum()
    # Setting the text to display the total reshare
    reshare_text = str(total_reshare)
else:
    # Setting the text to display 'Data Unavailable'
    reshare_text = 'Data Unavailable'

#Comments block
# Checking if the 'CommentCount' column exists in the DataFrame
if 'ReplyCount' in df.columns:
    # Calculating the sum of comments
    total_comment = df['ReplyCount'].sum()
    # Setting the text to display the total commments
    comment_text = str(total_comment)
else:
    # Setting the text to display 'Data Unavailable'
    comment_text = 'Data Unavailable'

#Viewers block
# Checking if the 'ViewersCount' column exists in the DataFrame
if 'ViewersCount' in df.columns:
    # Calculating the sum of viewers
    total_viewers = df['ViewersCount'].sum()
    # Setting the text to display the total commments
    viewers_text = str(total_viewers)
else:
    # Setting the text to display 'Data Unavailable'
    viewers_text = 'Data Unavailable'



