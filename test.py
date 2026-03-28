from url_classifier import predict

# Add any URLs you want to test here
test_cases = [
    ("https://youtube.com/watch?v=abc", "Python Tutorial for Beginners"),
    ("https://github.com/myproject", "My Deep Learning Project"),
    ("https://instagram.com/reels", "Trending Reels"),
    ("https://google.com/search?q=weather", "weather - Google Search"),
    ("https://netflix.com/watch", "Breaking Bad"),
    ("https://leetcode.com/problems/binary-search", "Binary Search"),
]

print("=== Productivity Predictions ===\n")
for url, title in test_cases:
    result = predict(url, title)
    bar = "█" * int(result["productive_pct"] / 10) + "░" * (10 - int(result["productive_pct"] / 10))
    print(f"{title}")
    print(f"  Label : {result['predicted_label'].upper()}")
    print(f"  Prod% : [{bar}] {result['productive_pct']}%")
    print()
