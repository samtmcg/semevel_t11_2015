def own_tokenizer(sent):
        # input comes in pre-tokenized, and tokens are sepparated by white space
        # this is used in the *Vectorizer functions
        return sent.split(' ')


def replace_user_tags(tweet):
        # removes references to other users, but replaces with a special token,
        # so does not remove the fact that they do reference others
        split_tweet = tweet.split(' ')
        nameless_tweet=[]
        for w in split_tweet:
                if w[0] == '@':
                        nameless_tweet.append('referenceAnotherUser')
                else:
                        nameless_tweet.append(w)
        fixed_tweet = (' ').join(nameless_tweet)
        return fixed_tweet

def remove_hash_tags(tweet):
	split_tweet = tweet.split(' ')
	wl = [w for w in split_tweet if not w[0] =='#']
	tweet = (' ').join(wl)
	return tweet
