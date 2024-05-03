from warnings import filterwarnings
filterwarnings('ignore')

# API BAĞLANTISININ YAPILMASI
import tweepy, codecs


consumer_key = '3FkYv8xAoq35ojMtSeIKeLsYo'
consumer_secret = '6p3oSAgDGROc1IcuGv6MDG67ZJF77TqSNis7PWskgDgT6JyRWy'
access_token = '1295062497540669447-3dRQrVrxk9fyVqNXQAeRfOOttnJEhN'
access_token_secret = 'qCA8jwjTcPNXaXgYmzYSxdBBWIozvAPNIWcx54NL548jx'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

api.update_status("Hello from Python for Udemy Data Science and Machine Learning Course")