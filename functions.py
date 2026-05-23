# functions.py
# For Contnet based filtering
import pandas as pd                          # For DataFrame operations
import numpy as np                           # For numerical operations (optional here but useful)

def recommend_content_based(title, top_n, movies, movie_idx, content_sim):   # Function definition with inputs
    if title not in movie_idx:                                               # Check if movie exists
        raise ValueError(f"Movie '{title}' not found.")                      # Raise error if not found

    idx = movie_idx[title]                                                   # Get index of selected movie

    sim_scores = list(enumerate(content_sim[idx]))                           # Get similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)        # Sort movies by similarity (highest first), x[0] -> index score and x[1] -> similarity score

    sim_scores = sim_scores[1: top_n + 1]                                    # Skip itself and take top N

    movie_indices = [i for i, score in sim_scores]                           # Extract indices
    scores = [score for i, score in sim_scores]                              # Extract scores

    recs = movies.iloc[movie_indices][["movieId", "title", "genres"]].copy() # Get movie details
    recs["similarity_score"] = scores                                        # Add similarity scores

    return recs.reset_index(drop=True)                                       # Return result

# For User - User Collaboration

def recommend_user_user(user_id, user_item, user_sim_df, movies, top_n = 10, min_sim = 0.05):   # Function for user-user collaborative filtering
    
    if user_id not in user_item.index:                                                      # Check if user exists in dataset
        raise ValueError(f"User {user_id} not found.")                                      # Raise error if user not found

    # Users similar to target user
    sims = user_sim_df.loc[user_id].drop(user_id, errors="ignore")                          # Get similarity scores for user and remove self
    sims = sims[sims > min_sim].sort_values(ascending=False)                                # Keep users above similarity threshold and sort

    if sims.empty:                                                                          # If no similar users found
        return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_rating"])     # Return empty DataFrame

    target_ratings = user_item.loc[user_id]                                                 # Get ratings of the target user
    unseen_movies = target_ratings[target_ratings.isna()].index                             # Find movies the user has not rated

    predictions = {}                                                                        # Dictionary to store predicted ratings

    for movie_id in unseen_movies:                                                          # Loop through each unseen movie
        neighbor_ratings = user_item.loc[sims.index, movie_id].dropna()                     # Get ratings from similar users for this movie
        
        if len(neighbor_ratings) == 0:                                                      # If no similar users rated it
            continue                                                                        # Skip this movie

        aligned_sims = sims.loc[neighbor_ratings.index]                                     # Match similarity scores with users who rated
        denom = aligned_sims.abs().sum()                                                    # Sum of similarity values (denominator)
        
        if denom == 0:                                                                      # Avoid division by zero
            continue                                                                        # Skip this movie

        pred = np.dot(neighbor_ratings.values, aligned_sims.values) / denom                 # Weighted average prediction
        predictions[movie_id] = pred                                                        # Store predicted rating

    if len(predictions) == 0:                                                               # If no predictions generated
        return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_rating"])     # Return empty DataFrame

    recs = (
        pd.DataFrame(predictions.items(), columns=["movieId", "predicted_rating"])         # Convert predictions dict to DataFrame
        .sort_values("predicted_rating", ascending=False)                                  # Sort by highest predicted rating
        .head(top_n)                                                                       # Select top N recommendations
    )

    recs = recs.merge(                                                                     # Add movie details
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return recs[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop = True)  # Return final result


# Item-item collaborative Filtering

def recommend_item_item(user_id, user_item, item_sim_df, movies, top_n = 10, min_user_rating = 4.0):   # Define item-item collaborative filtering function
    
    if user_id not in user_item.index:                                                                 # Check whether the given user exists
        raise ValueError(f"User {user_id} not found.")                                                 # Raise an error if user does not exist

    user_ratings = user_item.loc[user_id].dropna()                                                     # Get all ratings given by the target user

    liked_items = user_ratings[user_ratings >= min_user_rating]                                        # Keep only movies rated at or above the threshold
    unseen_items = user_item.loc[user_id][user_item.loc[user_id].isna()].index                         # Find movies the user has not rated yet

    if len(liked_items) == 0:                                                                          # If no liked items exist
        return pd.DataFrame(columns = ["movieId", "title", "genres", "predicted_rating"])              # Return empty DataFrame

    scores = {}                                                                                        # Create dictionary to store predicted scores

    for candidate in unseen_items:                                                                     # Loop through each unseen movie
        sim_sum = 0.0                                                                                  # Store total similarity for the candidate movie
        weighted_sum = 0.0                                                                             # Store weighted sum of similarity × rating

        for liked_movie, rating in liked_items.items():                                                # Loop through movies the user already liked
            
            if candidate not in item_sim_df.index or liked_movie not in item_sim_df.columns:           # Check if movie IDs exist in similarity matrix
                continue                                                                               # Skip if not present

            sim = item_sim_df.loc[candidate, liked_movie]                                              # Get similarity between unseen movie and liked movie
            
            if sim <= 0:                                                                               # Ignore zero or negative similarity
                continue                                                                               # Skip to the next liked movie

            weighted_sum += sim * rating                                                               # Add weighted contribution to total
            sim_sum += sim                                                                             # Add similarity to similarity total

        if sim_sum > 0:                                                                                # Check that denominator is not zero
            scores[candidate] = weighted_sum / sim_sum                                                 # Compute predicted score for the unseen movie

    if len(scores) == 0:                                                                               # If no scores were generated
        return pd.DataFrame(columns = ["movieId", "title", "genres", "predicted_rating"])              # Return empty DataFrame

    recs = (                                                                                           # Start building recommendations DataFrame
        pd.DataFrame(scores.items(), columns = ["movieId", "predicted_rating"])                        # Convert scores dictionary into DataFrame
        .sort_values("predicted_rating", ascending = False)                                            # Sort movies by predicted rating, highest first
        .head(top_n)                                                                                   # Keep only top N recommendations
    )

    recs = recs.merge(movies[["movieId", "title", "genres"]], on = "movieId", how = "left")            # Merge with movie details

    return recs[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop = True)           # Return final recommendations