#import nltk
import json
from transformers import pipeline
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

def main():
    with open('example_data.json', 'r') as file:
        data = json.load(file)

    reaction_sentiment_cache = {}

    # adjust these weights as needed
    # the bigger the weight, the bigger it's impact on channel sentiment analysis
    post_weight = 0.4
    comment_weight = 0.4
    reactions_weight = 0.2


    for channel, posts in data.items():
        print(f"channel: {channel}")

        posts_quantity = 0
        posts_sentiment_aggregated = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        comments_sentiment_aggregated = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        reactions_sentiment_aggregated = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for post in posts:

            posts_quantity += 1
            print(f"\nanalyzing text and reactions for post: {post['datetime']}")

            post_text = post['post_text']
            post_sentiment = analyze_sentiment(post_text)
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
                reaction_type = reaction['type']

                # cache reactions for better performance / unfortunately we cant cache messages ;( 
                if reaction_type not in reaction_sentiment_cache:
                    reaction_sentiment_cache[reaction_type] = analyze_sentiment(reaction_type)

                reaction_sentiment = reaction_sentiment_cache[reaction_type]
                reactions_sentiment_aggregated[reaction_sentiment[0]['label']] += reaction_sentiment[0]['score']

                print(f"reaction: {reaction_type} : {reaction['number_of_reactions']}; reaction sentiment: {reaction_sentiment}")
                
                label = reaction_sentiment[0]['label']
                sentiments[label] += reaction_sentiment[0]['score'] * reaction['number_of_reactions']

            
            # now push reaction_sentiment (likes/dislikes) for this post
            
            ratio_normalization(sentiments) # return ratio, push it too


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

        print(f"channel sentiment: {channel_sentiment}")

        # now push posts_quantity, channel_sentiment for this channel

if __name__ == "__main__":
    main()
