from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Combine the texts into a list
    texts = [text1, text2]

    # Fit and transform the texts into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity between the first and second text
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]

def main():
    # Example texts
    text1 = """Plagiarism is the practice of taking someone else's work or ideas and passing them off as your own."""
    text2 = """Plagiarism involves using someone else's work or ideas without proper attribution."""

    # Calculate the similarity
    similarity_score = calculate_similarity(text1, text2)
    print(f"Similarity score: {similarity_score:.4f}")

    # Determine if the texts are similar
    threshold = 0.5  # You can adjust this threshold based on your needs
    if similarity_score > threshold:
        print("The texts are similar.")
    else:
        print("The texts are not similar.")

if __name__ == "__main__":
    main()
