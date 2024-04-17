import pandas as pd
import random

def generate_user_name():
    adjectives = ["Happy", "Lucky", "Creative", "Adventurous", "Sunny", "Gentle", "Brave", "Clever", "Kind", "Wild"]
    nouns = ["Explorer", "Dreamer", "Traveler", "Artist", "Reader", "Foodie", "Admirer", "Seeker", "Wanderer", "Lover"]
    return f"{random.choice(adjectives)}{random.choice(nouns)}"

users_data = {
    'user_id': list(range(1, 51)),
    'user_name': [generate_user_name() for _ in range(50)]
}

users_df = pd.DataFrame(users_data)

posts_data = {
    'user_id': [random.randint(1, 50) for _ in range(50)],
    'item_id': [random.randint(1001, 1050) for _ in range(50)],
    'liked': [random.choice([0, 1]) for _ in range(50)],
    'title': [
        "Fashion Trends for This Season",
        "Delicious Recipes to Try at Home",
        "10 Home Decor Ideas to Beautify Your Space",
        "Top Travel Destinations for Adventure Seekers",
        "Exploring the World of Art and Design",
        "Healthy Living: Tips for a Balanced Lifestyle",
        "DIY Projects for Creative Minds",
        "The Ultimate Guide to Self-Care",
        "Inspiring Quotes to Brighten Your Day",
        "Capturing Moments: Photography Tips",
        "Discovering New Music: Must-Listen Tracks",
        "The Power of Meditation: Finding Inner Peace",
        "Mindful Eating: Nourishing Your Body and Soul",
        "Stay Fit and Active: Exercise Ideas",
        "Bookworm's Paradise: Must-Read Books",
        "Exploring Different Cultures: Cultural Festivals",
        "Gardening Tips for Green Thumbs",
        "Budget-Friendly Travel Tips",
        "Stunning Wedding Ideas for Your Special Day",
        "Unlocking Creativity: Artistic Inspiration",
        "Healthy Snack Ideas for On-the-Go",
        "Mindfulness in Daily Life: Simple Practices",
        "Fashionable Accessories for Every Occasion",
        "Delightful Desserts to Satisfy Your Sweet Tooth",
        "Interior Design Trends: Stylish Ideas",
        "Adventures in Nature: Outdoor Activities",
        "Crafting for Fun: DIY Projects",
        "Digital Nomad Lifestyle: Working Remotely",
        "Culinary Adventures: Exploring New Cuisines",
        "Inspirational Stories of Resilience",
        "Healthy Habits for a Happy Life",
        "Artisanal Crafts: Handmade Treasures",
        "Travel Photography: Capturing Memories",
        "Mindful Parenting: Nurturing Growth",
        "Creative Writing Prompts: Spark Your Imagination",
        "Eco-Friendly Living: Sustainable Practices",
        "Exploring the World of Fashion",
        "Tasty Breakfast Ideas to Start Your Day Right",
        "Home Office Essentials for Productivity",
        "Finding Balance: Work-Life Harmony",
        "Exploring Local Cuisine: Foodie Adventures",
        "Mindfulness Meditation: Calm Your Mind",
        "Outdoor Entertaining Ideas for Summer",
        "The Joy of Painting: Artistic Expression",
        "Discovering Hidden Gems: Travel Secrets",
        "Mindful Movement: Yoga for Wellness",
        "The Power of Positive Thinking",
        "Stylish Outfit Ideas for Every Season",
        "Healthy Skin Habits: Skincare Tips",
        "Sustainable Fashion: Ethical Choices"
    ]
}

posts_df = pd.DataFrame(posts_data)

# Generate tags for posts
tags = ["fashion", "food", "home", "travel", "art", "health", "DIY", "self-care", "inspiration", "photography"]
tag_data = {'tag_' + tag: [random.choice([0, 1]) for _ in range(50)] for tag in tags}
tags_df = pd.DataFrame(tag_data)

# Concatenate tags dataframe with posts dataframe
posts_df = pd.concat([posts_df, tags_df], axis=1)

# save data as csv files
users_df.to_csv('users_data.csv', index=False)
posts_df.to_csv('posts_data.csv', index=False)
