"""
User profile features extraction module.
Implements user engagement features as described in the paper.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger(__name__)

class UserProfileExtractor:
    """
    Extract user profile features from tweet data.
    Implements the 10-dimensional profile features mentioned in the paper.
    """
    
    def __init__(self):
        self.profile_features = [
            'followers_count',
            'friends_count', 
            'listed_count',
            'favourites_count',
            'statuses_count',
            'account_age_days',
            'followers_friends_ratio',
            'tweets_per_day',
            'verified_status',
            'has_description'
        ]
        logger.info("Initialized UserProfileExtractor")
    
    def extract_tweet_features(self, tweet_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from tweet-level data.
        
        Args:
            tweet_data: DataFrame with tweet information
            
        Returns:
            Dictionary of tweet-based features
        """
        features = {}
        
        if len(tweet_data) == 0:
            return {f'tweet_{feat}': 0.0 for feat in ['count', 'avg_length', 'has_hashtags', 'has_mentions', 'has_urls']}
        
        # Tweet count
        features['tweet_count'] = len(tweet_data)
        
        # Average tweet length
        if 'text' in tweet_data.columns:
            tweet_lengths = tweet_data['text'].fillna('').str.len()
            features['avg_tweet_length'] = tweet_lengths.mean()
        else:
            features['avg_tweet_length'] = 0.0
        
        # Hashtag usage
        if 'text' in tweet_data.columns:
            hashtag_count = tweet_data['text'].fillna('').str.count('#')
            features['has_hashtags'] = (hashtag_count > 0).mean()
        else:
            features['has_hashtags'] = 0.0
        
        # Mention usage
        if 'text' in tweet_data.columns:
            mention_count = tweet_data['text'].fillna('').str.count('@')
            features['has_mentions'] = (mention_count > 0).mean()
        else:
            features['has_mentions'] = 0.0
        
        # URL usage
        if 'text' in tweet_data.columns:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            url_count = tweet_data['text'].fillna('').str.count(url_pattern)
            features['has_urls'] = (url_count > 0).mean()
        else:
            features['has_urls'] = 0.0
        
        return features
    
    def extract_user_features(self, user_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from user profile data.
        
        Args:
            user_data: DataFrame with user profile information
            
        Returns:
            Dictionary of user-based features
        """
        features = {}
        
        if len(user_data) == 0:
            return {feat: 0.0 for feat in self.profile_features}
        
        # Basic profile features
        features['followers_count'] = user_data.get('followers', pd.Series([0])).iloc[0]
        features['friends_count'] = user_data.get('friends', pd.Series([0])).iloc[0]
        features['listed_count'] = user_data.get('listed', pd.Series([0])).iloc[0]
        features['favourites_count'] = user_data.get('favourites', pd.Series([0])).iloc[0]
        features['statuses_count'] = user_data.get('statuses', pd.Series([0])).iloc[0]
        
        # Account age
        if 'created_at' in user_data.columns:
            try:
                created_at = pd.to_datetime(user_data['created_at'].iloc[0])
                account_age = (datetime.now() - created_at).days
                features['account_age_days'] = max(0, account_age)
            except:
                features['account_age_days'] = 0
        else:
            features['account_age_days'] = 0
        
        # Followers to friends ratio
        if features['friends_count'] > 0:
            features['followers_friends_ratio'] = features['followers_count'] / features['friends_count']
        else:
            features['followers_friends_ratio'] = 0.0
        
        # Tweets per day
        if features['account_age_days'] > 0:
            features['tweets_per_day'] = features['statuses_count'] / features['account_age_days']
        else:
            features['tweets_per_day'] = 0.0
        
        # Verified status (simplified - check if user has high follower count)
        features['verified_status'] = 1.0 if features['followers_count'] > 10000 else 0.0
        
        # Has description (simplified - check if user has profile info)
        features['has_description'] = 1.0 if features['followers_count'] > 0 else 0.0
        
        return features
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features to 0-1 range for better model performance.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Normalized feature dictionary
        """
        normalized = {}
        
        # Define normalization ranges (based on typical Twitter data)
        ranges = {
            'followers_count': 1000000,
            'friends_count': 5000,
            'listed_count': 1000,
            'favourites_count': 10000,
            'statuses_count': 50000,
            'account_age_days': 3650,  # 10 years
            'followers_friends_ratio': 10.0,
            'tweets_per_day': 50.0,
            'verified_status': 1.0,
            'has_description': 1.0,
            'tweet_count': 100,
            'avg_tweet_length': 280,
            'has_hashtags': 1.0,
            'has_mentions': 1.0,
            'has_urls': 1.0
        }
        
        for key, value in features.items():
            if key in ranges:
                normalized[key] = min(1.0, max(0.0, value / ranges[key]))
            else:
                normalized[key] = value
        
        return normalized
    
    def extract_profile_features(self, 
                                news_data: pd.DataFrame,
                                tweet_data: Optional[pd.DataFrame] = None,
                                user_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract profile features for news articles.
        
        Args:
            news_data: DataFrame with news articles
            tweet_data: Optional DataFrame with tweet information
            user_data: Optional DataFrame with user profile information
            
        Returns:
            DataFrame with profile features added
        """
        logger.info(f"Extracting profile features for {len(news_data)} news articles...")
        
        result_data = news_data.copy()
        
        # Initialize profile feature columns
        for feat in self.profile_features:
            result_data[f'profile_{feat}'] = 0.0
        
        # Add tweet-based features
        tweet_features = ['tweet_count', 'avg_tweet_length', 'has_hashtags', 'has_mentions', 'has_urls']
        for feat in tweet_features:
            result_data[f'profile_{feat}'] = 0.0
        
        # Process each news article
        for idx, row in news_data.iterrows():
            features = {}
            
            # Extract tweet features if tweet data is available
            if tweet_data is not None and 'tweet_ids' in row:
                tweet_ids = str(row['tweet_ids']).split('\t')
                if len(tweet_ids) > 0 and tweet_ids[0] != 'nan':
                    # Filter tweets for this article
                    article_tweets = tweet_data[tweet_data['tweet_id'].astype(str).isin(tweet_ids)]
                    tweet_features_dict = self.extract_tweet_features(article_tweets)
                    features.update(tweet_features_dict)
            
            # Extract user features if user data is available
            if user_data is not None and 'tweet_ids' in row:
                tweet_ids = str(row['tweet_ids']).split('\t')
                if len(tweet_ids) > 0 and tweet_ids[0] != 'nan':
                    # Get users who tweeted about this article
                    if tweet_data is not None:
                        article_tweets = tweet_data[tweet_data['tweet_id'].astype(str).isin(tweet_ids)]
                        if len(article_tweets) > 0:
                            user_ids = article_tweets['user_id'].unique()
                            article_users = user_data[user_data['user_id'].astype(str).isin(user_ids)]
                            if len(article_users) > 0:
                                user_features_dict = self.extract_user_features(article_users)
                                features.update(user_features_dict)
            
            # Normalize features
            features = self.normalize_features(features)
            
            # Update result dataframe
            for key, value in features.items():
                if f'profile_{key}' in result_data.columns:
                    result_data.at[idx, f'profile_{key}'] = value
        
        logger.info(f"Completed profile feature extraction")
        return result_data


def create_synthetic_user_data(n_users: int = 100) -> pd.DataFrame:
    """
    Create synthetic user data for testing.
    
    Args:
        n_users: Number of users to create
        
    Returns:
        DataFrame with synthetic user data
    """
    logger.info(f"Creating synthetic user data for {n_users} users...")
    
    data = []
    for i in range(n_users):
        # Create realistic user profiles
        followers = np.random.lognormal(mean=6, sigma=2)  # Log-normal distribution
        friends = np.random.lognormal(mean=5, sigma=1.5)
        
        data.append({
            'user_id': f'user_{i}',
            'screen_name': f'user{i}',
            'followers': int(followers),
            'friends': int(friends),
            'listed': int(np.random.lognormal(mean=2, sigma=1)),
            'favourites': int(np.random.lognormal(mean=4, sigma=1.5)),
            'statuses': int(np.random.lognormal(mean=4, sigma=1.5)),
            'created_at': '2010-01-01',  # Simplified
            'verified': np.random.choice([True, False], p=[0.1, 0.9])
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic user data: {len(df)} users")
    return df


def create_synthetic_tweet_data(n_tweets: int = 1000) -> pd.DataFrame:
    """
    Create synthetic tweet data for testing.
    
    Args:
        n_tweets: Number of tweets to create
        
    Returns:
        DataFrame with synthetic tweet data
    """
    logger.info(f"Creating synthetic tweet data for {n_tweets} tweets...")
    
    data = []
    for i in range(n_tweets):
        # Create realistic tweet content
        content = f"This is tweet {i} about some news article. #news #breaking"
        if i % 3 == 0:
            content += " @someone"
        if i % 5 == 0:
            content += " https://example.com"
        
        data.append({
            'tweet_id': f'tweet_{i}',
            'user_id': f'user_{i % 50}',  # 50 different users
            'created_at': '2023-01-01',  # Simplified
            'text': content,
            'retweet_of': None,
            'in_reply_to': None,
            'lang': 'en'
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic tweet data: {len(df)} tweets")
    return df


if __name__ == "__main__":
    # Test the profile extractor
    extractor = UserProfileExtractor()
    
    # Create synthetic data
    user_data = create_synthetic_user_data(10)
    tweet_data = create_synthetic_tweet_data(50)
    
    # Create sample news data
    news_data = pd.DataFrame({
        'id': ['news_1', 'news_2'],
        'title': ['Test News 1', 'Test News 2'],
        'tweet_ids': ['tweet_0\ttweet_1', 'tweet_2\ttweet_3']
    })
    
    # Extract profile features
    result = extractor.extract_profile_features(news_data, tweet_data, user_data)
    
    print("Profile features extracted:")
    print(result[['id', 'title'] + [col for col in result.columns if col.startswith('profile_')]].head())
