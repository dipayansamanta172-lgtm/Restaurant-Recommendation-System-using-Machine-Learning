import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


DATASET_FILE = "Dataset.csv"
OUTPUT_FILE = "output.txt"
GRAPH_FILE = "recommendation_analysis_graph.png"


if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError("Dataset.csv must be in the same folder as task2.py")


def find_column(data, expected_name):
    for col in data.columns:
        if col.strip().lower() == expected_name.strip().lower():
            return col
    return None


def clean_text(value):
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().lower()
    value = " ".join(value.split())
    return value if value else "unknown"


def split_cuisines(value):
    if pd.isna(value):
        return []
    cuisines = []
    for cuisine in str(value).split(","):
        cuisine = cuisine.strip()
        if cuisine:
            cuisines.append(cuisine)
    return cuisines


restaurants = pd.read_csv(DATASET_FILE)

restaurant_name_col = find_column(restaurants, "Restaurant Name")
cuisines_col = find_column(restaurants, "Cuisines")
city_col = find_column(restaurants, "City")
price_col = find_column(restaurants, "Price range")
rating_col = find_column(restaurants, "Aggregate rating")
online_delivery_col = find_column(restaurants, "Has Online delivery")
table_booking_col = find_column(restaurants, "Has Table booking")

required_columns = [cuisines_col, city_col, price_col, rating_col]
if any(col is None for col in required_columns):
    raise ValueError("Dataset.csv must contain Cuisines, City, Price range, and Aggregate rating columns")

rows_before_preprocessing = restaurants.shape[0]
duplicate_rows = restaurants.duplicated().sum()
missing_values = restaurants.isna().sum()

model_data = restaurants.drop_duplicates().reset_index(drop=True).copy()
model_data[cuisines_col] = model_data[cuisines_col].apply(clean_text)
model_data[city_col] = model_data[city_col].apply(clean_text)

price_numbers = pd.to_numeric(model_data[price_col], errors="coerce")
rating_numbers = pd.to_numeric(model_data[rating_col], errors="coerce")
model_data[price_col] = price_numbers.fillna(price_numbers.median()).fillna(0)
model_data[rating_col] = rating_numbers.fillna(rating_numbers.median()).fillna(0)

if online_delivery_col is not None:
    model_data[online_delivery_col] = model_data[online_delivery_col].apply(clean_text)
if table_booking_col is not None:
    model_data[table_booking_col] = model_data[table_booking_col].apply(clean_text)

all_cuisines = model_data[cuisines_col].apply(split_cuisines).explode().dropna()
original_cuisines = restaurants[cuisines_col].fillna("Unknown").apply(split_cuisines).explode().dropna()

cuisine_ratings = model_data[[cuisines_col, rating_col]].copy()
cuisine_ratings["Cuisine"] = cuisine_ratings[cuisines_col].apply(split_cuisines)
cuisine_ratings = cuisine_ratings.explode("Cuisine")

recommendation_text = model_data[cuisines_col].str.replace(",", " ", regex=False)
recommendation_text = recommendation_text + " city_" + model_data[city_col].str.replace(" ", "_", regex=False)

features_used = ["Cuisines", "City", "Price range", "Aggregate rating"]
if online_delivery_col is not None:
    recommendation_text = recommendation_text + " online_delivery_" + model_data[online_delivery_col].str.replace(" ", "_", regex=False)
    features_used.append("Has Online delivery")
if table_booking_col is not None:
    recommendation_text = recommendation_text + " table_booking_" + model_data[table_booking_col].str.replace(" ", "_", regex=False)
    features_used.append("Has Table booking")

vectorizer = TfidfVectorizer(stop_words="english")
text_features = vectorizer.fit_transform(recommendation_text).toarray()

numeric_features = model_data[[price_col, rating_col]].astype(float)
scaler = StandardScaler()
scaled_numeric_features = scaler.fit_transform(numeric_features)

feature_matrix = np.hstack([text_features, scaled_numeric_features])
similarity_matrix = cosine_similarity(feature_matrix)

if len(model_data) <= 4:
    example_indexes = list(range(len(model_data)))
else:
    example_indexes = list(np.linspace(0, len(model_data) - 1, 4, dtype=int))

output_lines = []
output_lines.append("Dataset Overview")
output_lines.append(f"Rows before preprocessing: {rows_before_preprocessing}")
output_lines.append(f"Rows after preprocessing: {model_data.shape[0]}")
output_lines.append(f"Columns: {restaurants.shape[1]}")
output_lines.append(f"Duplicate rows: {duplicate_rows}")
output_lines.append("Missing values:")
for col, count in missing_values.items():
    output_lines.append(f"- {col}: {count}")

output_lines.append("")
output_lines.append("Exploratory Analysis Findings")
output_lines.append(f"Number of unique cuisines: {all_cuisines.nunique()}")
output_lines.append(f"Number of unique cuisine combinations: {model_data[cuisines_col].nunique()}")
output_lines.append(f"Number of cities: {model_data[city_col].nunique()}")
price_ranges = sorted(model_data[price_col].dropna().unique())
output_lines.append(f"Price ranges available: {', '.join(str(price) for price in price_ranges)}")
output_lines.append(f"Average aggregate rating: {model_data[rating_col].mean():.2f}")
output_lines.append(f"Minimum aggregate rating: {model_data[rating_col].min():.2f}")
output_lines.append(f"Maximum aggregate rating: {model_data[rating_col].max():.2f}")

output_lines.append("")
output_lines.append("Cuisine Analysis")
output_lines.append("Most common cuisines:")
for cuisine, count in all_cuisines.value_counts().head(10).items():
    output_lines.append(f"- {cuisine}: {count}")
output_lines.append("Most common cuisine combinations:")
for cuisine, count in model_data[cuisines_col].value_counts().head(10).items():
    output_lines.append(f"- {cuisine}: {count}")
output_lines.append("Average rating by cuisine:")
for cuisine, rating in cuisine_ratings.groupby("Cuisine")[rating_col].mean().sort_values(ascending=False).head(10).items():
    output_lines.append(f"- {cuisine}: {rating:.2f}")
output_lines.append("Restaurant distribution by cuisine:")
for cuisine, count in all_cuisines.value_counts().head(15).items():
    output_lines.append(f"- {cuisine}: {count}")

output_lines.append("")
output_lines.append("City Analysis")
output_lines.append("Average rating by city:")
for city, rating in model_data.groupby(city_col)[rating_col].mean().sort_values(ascending=False).head(10).items():
    output_lines.append(f"- {city}: {rating:.2f}")
output_lines.append("Restaurant distribution by city:")
for city, count in model_data[city_col].value_counts().head(15).items():
    output_lines.append(f"- {city}: {count}")

output_lines.append("")
output_lines.append("Price Range Analysis")
output_lines.append("Average rating by price range:")
for price_range, rating in model_data.groupby(price_col)[rating_col].mean().sort_index().items():
    output_lines.append(f"- Price range {price_range}: {rating:.2f}")
output_lines.append("Restaurant distribution by price range:")
for price_range, count in model_data[price_col].value_counts().sort_index().items():
    output_lines.append(f"- Price range {price_range}: {count}")

output_lines.append("")
output_lines.append("Additional Dataset Analysis")
output_lines.append("Top 10 cities by restaurant count:")
for city, count in model_data[city_col].value_counts().head(10).items():
    output_lines.append(f"- {city}: {count}")
output_lines.append("Top 10 cuisines by restaurant count:")
for cuisine, count in all_cuisines.value_counts().head(10).items():
    output_lines.append(f"- {cuisine}: {count}")
output_lines.append("Top 10 cuisine combinations by restaurant count:")
for cuisine, count in model_data[cuisines_col].value_counts().head(10).items():
    output_lines.append(f"- {cuisine}: {count}")
output_lines.append("Average rating by city:")
for city, rating in model_data.groupby(city_col)[rating_col].mean().sort_values(ascending=False).head(10).items():
    output_lines.append(f"- {city}: {rating:.2f}")
output_lines.append("Average rating by cuisine:")
for cuisine, rating in cuisine_ratings.groupby("Cuisine")[rating_col].mean().sort_values(ascending=False).head(10).items():
    output_lines.append(f"- {cuisine}: {rating:.2f}")
output_lines.append("Average rating by price range:")
for price_range, rating in model_data.groupby(price_col)[rating_col].mean().sort_index().items():
    output_lines.append(f"- Price range {price_range}: {rating:.2f}")

if online_delivery_col is not None:
    output_lines.append("Average rating by online delivery availability:")
    for value, rating in model_data.groupby(online_delivery_col)[rating_col].mean().sort_index().items():
        output_lines.append(f"- {value}: {rating:.2f}")
else:
    output_lines.append("Average rating by online delivery availability: Column not available")

if table_booking_col is not None:
    output_lines.append("Average rating by table booking availability:")
    for value, rating in model_data.groupby(table_booking_col)[rating_col].mean().sort_index().items():
        output_lines.append(f"- {value}: {rating:.2f}")
else:
    output_lines.append("Average rating by table booking availability: Column not available")

output_lines.append("Distribution of restaurants across price ranges:")
for price_range, count in model_data[price_col].value_counts().sort_index().items():
    output_lines.append(f"- Price range {price_range}: {count}")

if online_delivery_col is not None:
    online_count = (model_data[online_delivery_col] == "yes").sum()
    online_percentage = (online_count / len(model_data)) * 100 if len(model_data) > 0 else 0
    output_lines.append(f"Restaurants with online delivery: {online_count} ({online_percentage:.2f}%)")
else:
    output_lines.append("Restaurants with online delivery: Column not available")

if table_booking_col is not None:
    booking_count = (model_data[table_booking_col] == "yes").sum()
    booking_percentage = (booking_count / len(model_data)) * 100 if len(model_data) > 0 else 0
    output_lines.append(f"Restaurants with table booking: {booking_count} ({booking_percentage:.2f}%)")
else:
    output_lines.append("Restaurants with table booking: Column not available")

high_rating_count = (model_data[rating_col] > 4.0).sum()
high_rating_percentage = (high_rating_count / len(model_data)) * 100 if len(model_data) > 0 else 0
output_lines.append(f"Restaurants with aggregate rating greater than 4.0: {high_rating_count} ({high_rating_percentage:.2f}%)")
output_lines.append("Highest rated cuisine categories:")
for cuisine, rating in cuisine_ratings.groupby("Cuisine")[rating_col].mean().sort_values(ascending=False).head(10).items():
    output_lines.append(f"- {cuisine}: {rating:.2f}")
output_lines.append("Lowest rated cuisine categories:")
for cuisine, rating in cuisine_ratings.groupby("Cuisine")[rating_col].mean().sort_values(ascending=True).head(10).items():
    output_lines.append(f"- {cuisine}: {rating:.2f}")
output_lines.append("Highest rated cities:")
for city, rating in model_data.groupby(city_col)[rating_col].mean().sort_values(ascending=False).head(10).items():
    output_lines.append(f"- {city}: {rating:.2f}")
output_lines.append("Lowest rated cities:")
for city, rating in model_data.groupby(city_col)[rating_col].mean().sort_values(ascending=True).head(10).items():
    output_lines.append(f"- {city}: {rating:.2f}")

output_lines.append("")
output_lines.append("Recommendation Model Information")
output_lines.append("Recommendation approach used: Content-based filtering with TF-IDF vectorization, StandardScaler, and cosine similarity")
output_lines.append(f"Features used: {', '.join(features_used)}")
output_lines.append(f"Number of restaurants considered: {model_data.shape[0]}")
output_lines.append(f"Number of restaurants used after preprocessing: {model_data.shape[0]}")
output_lines.append(f"Number of unique cuisines: {all_cuisines.nunique()}")
output_lines.append(f"Number of cities: {model_data[city_col].nunique()}")
output_lines.append(f"Number of recommendation features generated: {feature_matrix.shape[1]}")

output_lines.append("")
output_lines.append("Sample Recommendations")
quality_rows = []
all_recommendation_scores = []
for restaurant_index in example_indexes:
    restaurant = model_data.iloc[restaurant_index]
    restaurant_name = restaurant[restaurant_name_col] if restaurant_name_col is not None else f"Restaurant index {restaurant_index}"
    scores = list(enumerate(similarity_matrix[restaurant_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scores = [item for item in scores if item[0] != restaurant_index]
    scores = scores[:5]
    all_recommendation_scores.extend([score for _, score in scores])

    output_lines.append(f"Restaurant Name: {restaurant_name}")
    output_lines.append("Recommended Restaurants:")
    for matched_index, score in scores:
        matched_restaurant = model_data.iloc[matched_index]
        matched_name = matched_restaurant[restaurant_name_col] if restaurant_name_col is not None else f"Restaurant index {matched_index}"
        output_lines.append(
            f"- {matched_name} | City: {matched_restaurant[city_col]} | Cuisines: {matched_restaurant[cuisines_col]} | "
            f"Price Range: {matched_restaurant[price_col]} | Aggregate Rating: {matched_restaurant[rating_col]:.2f} | "
            f"Similarity Score: {score:.4f}"
        )
    output_lines.append("")
    quality_rows.append((restaurant_index, restaurant_name, scores))

output_lines.append("Recommendation Metrics")
if all_recommendation_scores:
    output_lines.append(f"Average similarity score across all generated recommendations: {np.mean(all_recommendation_scores):.4f}")
    output_lines.append(f"Highest similarity score observed: {np.max(all_recommendation_scores):.4f}")
    output_lines.append(f"Lowest similarity score observed: {np.min(all_recommendation_scores):.4f}")
else:
    output_lines.append("Average similarity score across all generated recommendations: 0.0000")
    output_lines.append("Highest similarity score observed: 0.0000")
    output_lines.append("Lowest similarity score observed: 0.0000")
output_lines.append(f"Number of recommendation features generated: {feature_matrix.shape[1]}")
output_lines.append(f"Number of restaurants used after preprocessing: {model_data.shape[0]}")
output_lines.append(f"Number of unique cuisines: {all_cuisines.nunique()}")
output_lines.append(f"Number of unique cuisine combinations: {model_data[cuisines_col].nunique()}")
output_lines.append(f"Number of unique cities: {model_data[city_col].nunique()}")
for restaurant_index, restaurant_name, scores in quality_rows:
    restaurant_scores = [score for _, score in scores]
    average_score = np.mean(restaurant_scores) if restaurant_scores else 0
    output_lines.append(f"Restaurant Name: {restaurant_name}")
    output_lines.append(f"- Average similarity score: {average_score:.4f}")
    output_lines.append(f"- Number of recommendations generated: {len(scores)}")

output_lines.append("")
output_lines.append("Recommendation Quality Checks")
for restaurant_index, restaurant_name, scores in quality_rows:
    restaurant = model_data.iloc[restaurant_index]
    if not scores:
        output_lines.append(f"Restaurant Name: {restaurant_name}")
        output_lines.append("- Recommendations available: 0")
        continue

    base_cuisines = set(split_cuisines(restaurant[cuisines_col]))
    shared_cuisine_count = 0
    same_price_count = 0
    same_city_count = 0
    rating_differences = []
    price_differences = []

    for matched_index, _ in scores:
        matched_restaurant = model_data.iloc[matched_index]
        matched_cuisines = set(split_cuisines(matched_restaurant[cuisines_col]))
        if base_cuisines.intersection(matched_cuisines):
            shared_cuisine_count += 1
        if restaurant[price_col] == matched_restaurant[price_col]:
            same_price_count += 1
        if restaurant[city_col] == matched_restaurant[city_col]:
            same_city_count += 1
        rating_differences.append(abs(float(restaurant[rating_col]) - float(matched_restaurant[rating_col])))
        price_differences.append(abs(float(restaurant[price_col]) - float(matched_restaurant[price_col])))

    output_lines.append(f"Restaurant Name: {restaurant_name}")
    output_lines.append(f"- Recommendations checked: {len(scores)}")
    output_lines.append(f"- Recommendations sharing at least one cuisine: {shared_cuisine_count}")
    output_lines.append(f"- Recommendations sharing the same price range: {same_price_count}")
    output_lines.append(f"- Recommendations from the same city: {same_city_count}")
    output_lines.append(f"- Average rating difference: {np.mean(rating_differences):.2f}")
    output_lines.append(f"- Average price range difference: {np.mean(price_differences):.2f}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
    output_file.write("\n".join(output_lines))

cuisine_counts = original_cuisines.value_counts().head(10)
plt.figure(figsize=(10, 6))
cuisine_counts.sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 10 Most Common Cuisines")
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisine")
plt.tight_layout()
plt.savefig(GRAPH_FILE)
plt.close()

print("output.txt created")
print("recommendation_analysis_graph.png created")
