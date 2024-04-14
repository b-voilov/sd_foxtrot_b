#import nltk
import json
from transformers import pipeline
import postgres
from psycopg2.extras import Json 
import os

analyze_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment") 
# paper: https://arxiv.org/abs/2104.12250

# WIP!!
# TODO: 
#   Analyze comments in batches rather than one by one, we may lose some accuracy tho
#   Parallelize the code
#   Find some faster models in: https://huggingface.co/models
#   Write in DynamoDB
#   May add number of reactions per post, but, so far it's replaced with reactions sentiment (both per post and per channel)


# normalize the feedback ratio for additional stats
def ratio_normalization(sentiments):
    total = sentiments['positive'] + sentiments['negative'] + sentiments['neutral']
    if total != 0:
        positives_percentage = round((sentiments['positive'] / total) * 100, 3)
        negatives_percentage = round((sentiments['negative'] / total) * 100, 3)
        neutrals_percentage = round((sentiments['neutral'] / total) * 100, 3)

        print(f"positives: {positives_percentage}; negatives {negatives_percentage}; neutrals: {neutrals_percentage}")
        #return positives_percentage, negatives_percentage, neutrals_percentage
        return {"positive": positives_percentage, "negative": negatives_percentage, "neutral": neutrals_percentage} #sentiments


def check_comment(comment):
    comment_text = comment['text']
    adapt_comment_text = comment_text.replace("'", "''")
    comments_data = postgres.get_comments_data(adapt_comment_text)
    if len(comments_data) > 0:
        comments_statistics = comments_data[0][2]
        print(comments_data)
        print(comments_statistics)
    else:
        comments_statistics = {}
    comment_users = []
    user_data = comment["from_user"]
    user_id = user_data["uid"]

    # check if there was a comment like that before
    if comment_text in comments_statistics:
        comments_statistics["comment_count"] += 1
        comment_users = comments_statistics["users"]
        if not (user_id in comment_users):
            comments_statistics["users"].append(user_id)
        print(comments_statistics)
        comments_statistics = Json(comments_statistics)
        postgres.update_comments(adapt_comment_text, comments_statistics)
    else:
        comments_statistics["comment_count"] = 1
        comments_statistics["users"] = [user_id]
        print(comments_statistics)
        comments_statistics = Json(comments_statistics)
        postgres.add_comments(adapt_comment_text, comments_statistics)



def analyse(current_file):
    path = "src/"+ current_file
    with open(path, 'r', encoding="utf-8") as file:
       # print(file.read())
        data = json.load(file)

    reaction_sentiment_cache = {}

    # adjust these weights as needed
    # the bigger the weight, the bigger it's impact on channel sentiment analysis
    post_weight = 0.4
    comment_weight = 0.4
    reactions_weight = 0.2

    for item in data:
        channel_id = item['id']
        channel = item['title']
        posts = item['posts']
        print(f"channel: {channel}")

        posts_quantity = 0
        posts_sentiment_aggregated = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        comments_sentiment_aggregated = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        reactions_sentiment_aggregated = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        if not (postgres.exist_channel(channel_id)):
            postgres.add_channel(channel_id, "Telegram", channel, posts_quantity) #TYPE CHECK

        for post in posts:
            posts_quantity += 1
            post_id = post['post_id']
            post_text = post['text']
            if not post_text == None:
                adapt_post_text = post_text.replace("'", "''")
            else: 
                post_text = ""
                adapt_post_text = post_text

            if not (postgres.exist_post(channel_id, post_id)):
                post_sentiment = analyze_sentiment(post_text)
                print(f"\nanalyzing text and reactions for post: {post['datetime']}")
                posts_sentiment_aggregated[post_sentiment[0]['label']] += post_sentiment[0]['score'] # <- here,
                # if we'll keep adding new sentiments forever, at some point we'll obviously exceed the
                # bounds of float in python, but, considering that float is 64bit by default, and considering our
                # not big enough and finite analysis -- it's extremely unlikely that we will ever exceed these bounds

                # push post_sentiment

                print(f"post sentiment: {post_sentiment}")
                
                sentiments = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                comments_quantity = 0

                for comment in post.get('comments', []):
                    comments_quantity += 1
                    check_comment(comment)

                    comment_text = comment['text']
                    comment_sentiment = analyze_sentiment(comment_text)
                    comments_sentiment_aggregated[comment_sentiment[0]['label']] += comment_sentiment[0]['score']
                    print(f"comment sentiment: {comment_sentiment}")

                    label = comment_sentiment[0]['label']
                    sentiments[label] += comment_sentiment[0]['score']

                    # now push comments_quantity for this post
                
                ratio_normalization(sentiments) # push sentiment ratio for comments now


                sentiments = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                reactions = post['reactions'][:5] # analyzing only the first 5 reactions, which is enough for our estimate

                for reaction in reactions:
                    reaction_type = reaction['emoticon']

                    # cache reactions for better performance / unfortunately we cant cache messages ;( 
                    if reaction_type not in reaction_sentiment_cache:
                        reaction_sentiment_cache[reaction_type] = analyze_sentiment(reaction_type)

                    reaction_sentiment = reaction_sentiment_cache[reaction_type]
                    reactions_sentiment_aggregated[reaction_sentiment[0]['label']] += reaction_sentiment[0]['score']

                    print(f"reaction: {reaction_type} : {reaction['count']}; reaction sentiment: {reaction_sentiment}")
                    
                    label = reaction_sentiment[0]['label']
                    sentiments[label] += reaction_sentiment[0]['score'] * reaction['count']

                all_reactions = Json(post["reactions"])
                postgres.add_post(channel_id, post_id, adapt_post_text, post['datetime'], post['media_in_post'], 
                                  comments_quantity, post['views'], all_reactions)
                
                # now push reaction_sentiment (likes/dislikes) for this post
                
                ratio_normalization(sentiments) # return ratio, push it too
            else:
                print(f'Post {post_id} exists. Skipping')


        # if posts_sentiment_aggregated["positive"] is None:
        #     posts_sentiment_aggregated["positive"] = 0
        
        # if posts_sentiment_aggregated["negative"] is None:
        #     posts_sentiment_aggregated["negative"] = 0
        
        # if posts_sentiment_aggregated["neutral"] is None:
        #     posts_sentiment_aggregated["neutral"] = 0

        # if comments_sentiment_aggregated["positive"] is None:
        #     comments_sentiment_aggregated["positive"] = 0
        
        # if comments_sentiment_aggregated["negative"] is None:
        #     comments_sentiment_aggregated["negative"] = 0
        
        # if comments_sentiment_aggregated["neutral"] is None:
        #     comments_sentiment_aggregated["neutral"] = 0

        # if reactions_sentiment_aggregated["positive"] is None:
        #     reactions_sentiment_aggregated["positive"] = 0
        
        # if reactions_sentiment_aggregated["negative"] is None:
        #     reactions_sentiment_aggregated["negative"] = 0
        
        # if reactions_sentiment_aggregated["neutral"] is None:
        #     reactions_sentiment_aggregated["neutral"] = 0

        channel_sentiment = {
            "positive": (
                round( ratio_normalization( posts_sentiment_aggregated )["positive"] * post_weight +
                ratio_normalization( comments_sentiment_aggregated )["positive"] * comment_weight +
                ratio_normalization( reactions_sentiment_aggregated )["positive"] * reactions_weight, 3)
            ),
            "negative": (
                round( ratio_normalization( posts_sentiment_aggregated )["negative"] * post_weight +
                ratio_normalization( comments_sentiment_aggregated )["negative"] * comment_weight +
                ratio_normalization( reactions_sentiment_aggregated )["negative"] * reactions_weight, 3)
            ),
            "neutral": (
                round( ratio_normalization( posts_sentiment_aggregated )["neutral"] * post_weight +
                ratio_normalization( comments_sentiment_aggregated )["neutral"] * comment_weight +
                ratio_normalization( reactions_sentiment_aggregated )["neutral"] * reactions_weight, 3)
            ),
        }

        postgres.update_channel(channel_id, posts_quantity, channel_sentiment)

        print(f"channel sentiment: {channel_sentiment}")

        # now push posts_quantity, channel_sentiment for this channel


def main():
    postgres.create_tables()
    arr = os.listdir("src")
    for file in arr:
        analyse(file)
     
if __name__ == "__main__":
    main()
