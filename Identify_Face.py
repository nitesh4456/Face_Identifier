import pandas as pd
import numpy as np
import face_recognition
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create per-person clusters
def create_per_person_clusters(df):
    embedding_columns = [f'embed_{i}' for i in range(128)]
    person_centroids = {}
    person_kmeans_models = {}

    for person in df['person'].unique():
        person_data = df[df['person'] == person][embedding_columns].values

        # Force exactly 1 cluster per person
        n_clusters = 1

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(person_data)
        centroids = kmeans.cluster_centers_

        person_centroids[person] = centroids
        person_kmeans_models[person] = kmeans

    return person_centroids, person_kmeans_models


# Match input embedding to best cluster centroid
def find_best_match(input_embedding, person_centroids):
    best_person = None
    best_distance = float('inf')

    for person, centroids in person_centroids.items():
        distances = np.linalg.norm(centroids - input_embedding, axis=1)
        min_distance = np.min(distances)

        if min_distance < best_distance:
            best_distance = min_distance
            best_person = person

    return best_person, best_distance

# Plot cluster distribution
def plot_cluster_distribution(person_centroids):
    persons = list(person_centroids.keys())
    num_clusters = [centroids.shape[0] for centroids in person_centroids.values()]

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        num_clusters,
        labels=persons,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.tab20(np.linspace(0, 1, len(persons))),
        explode=[0.05]*len(persons),
        textprops={'fontsize': 12}
    )

    plt.title('Distribution of Clusters by Person', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Plot clusters using PCA (with input image X marker)
def plot_clusters_pca(df, person_kmeans_models, input_embedding=None):
    embedding_columns = [f'embed_{i}' for i in range(128)]
    known_embeddings = df[embedding_columns].values
    persons = df['person'].values

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(known_embeddings)

    plt.figure(figsize=(12, 8))
    unique_persons = np.unique(persons)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_persons)))

    # Map person to color
    person_to_color = {person: colors[idx] for idx, person in enumerate(unique_persons)}

    for person in unique_persons:
        mask = (persons == person)
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            label=str(person),
            color=person_to_color[person],
            alpha=0.5,
            s=40
        )

        # Plot cluster centers
        kmeans = person_kmeans_models[person]
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            marker='X',
            color=person_to_color[person],  # <<< USE SAME COLOR
            s=200,
            edgecolor='black'
        )

    # Plot input image embedding if provided
    if input_embedding is not None:
        input_reduced = pca.transform(input_embedding.reshape(1, -1))
        plt.scatter(
            input_reduced[0, 0],
            input_reduced[0, 1],
            marker='X',
            color='red',
            s=300,
            edgecolor='black',
            label='Input Image'
        )

    plt.title('Clusters and Cluster Centers by Person (PCA Reduced)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    plt.show()

# Mean feature values for each cluster
def show_mean_features(person_centroids):
    print("\n=== Mean Feature Values of Each Cluster (shortened view) ===\n")
    for person, centroids in person_centroids.items():
        print(f"Person: {person}")
        for idx, center in enumerate(centroids):
            print(f"  Cluster {idx+1} mean embedding (first 5 dims): {center[:5]}")
        print("")



def visualize_mean_features(person_centroids):
    print("\n=== Visualizing Mean Feature Values for All Clusters and Persons ===\n")

    all_data = []
    all_labels = []

    for person, centroids in person_centroids.items():
        for idx, centroid in enumerate(centroids):
            all_data.append(centroid)
            all_labels.append(f'{person}_C{idx+1}')

    all_data = np.array(all_data)

    plt.figure(figsize=(20, 8))

    for i in range(len(all_data)):
        plt.plot(
            range(128),
            all_data[i],
            label=all_labels[i],
            marker='o',
            markersize=2,
            linewidth=1
        )

    plt.title('Mean Embedding Features Across All Persons and Clusters', fontsize=18)
    plt.xlabel('Embedding Feature Index (0-127)')
    plt.ylabel('Mean Value')
    plt.xticks(ticks=range(0, 128, 5), labels=[str(i) for i in range(0, 128, 5)], rotation=90)
    plt.grid(True)
    plt.legend(title='Person_Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    plt.show()


# def suggest_threshold(distances, percentile=95):
#     return np.percentile(distances, percentile)

def euclidean_to_cosine_similarity_percent(embedding, centroid):
    """
    Calculates cosine similarity % from Euclidean distance between two embeddings.

    Args:
        embedding (numpy array): Embedding of the input image.
        centroid (numpy array): Centroid of the cluster.

    Returns:
        float: Cosine similarity percentage (0-100%).
    """
    # First, normalize both vectors
    embedding_norm = embedding / np.linalg.norm(embedding)
    centroid_norm = centroid / np.linalg.norm(centroid)

    # Calculate Euclidean distance between normalized vectors
    euclidean_distance = np.linalg.norm(embedding_norm - centroid_norm)

    # Approximate cosine similarity from Euclidean distance
    cosine_similarity = 1 - (euclidean_distance ** 2) / 2

    # Convert to percentage
    cosine_similarity_percent = cosine_similarity * 100

    return cosine_similarity_percent


# Main Function
def find_most_similar(input_image_path, csv_path, top_k=5):
    # Load input image and compute embedding
    image = face_recognition.load_image_file(input_image_path)
    encodings = face_recognition.face_encodings(image)

    #print embeddings (i.e encodings)
    if len(encodings)==0:
        print("No face found or No embeddings recognised in input image. Try with different Image!")
        return
    print("The image embeddings are : ")
    for ele in encodings :
        print(ele)
    p=input("press any key to continue ")        

    input_embedding = encodings[0]
    # Load embeddings database
    df = pd.read_csv(csv_path)

    # Create per-person clusters
    person_centroids, person_kmeans_models = create_per_person_clusters(df)

    # Visualize clusters
    plot_cluster_distribution(person_centroids)
    plot_clusters_pca(df, person_kmeans_models, input_embedding)
    # show_mean_features(person_centroids)
    visualize_mean_features(person_centroids)

    # Match using cluster centroids
    best_person, best_distance = find_best_match(input_embedding, person_centroids)

    # Calculate Cosine Similarity Percentage
    best_centroid = person_centroids[best_person][0]  # assuming 1 cluster per person
    similarity_percent = euclidean_to_cosine_similarity_percent(input_embedding, best_centroid)

    print("\n=== Cluster-Based Match ===")
    print(f"Predicted Person: {best_person} | Distance to Closest Cluster Center: {best_distance:.5f}")
    print(f"Cosine Similarity (approx.) ,Accuracy : {similarity_percent:.2f}%")

    # Compute normal top-k based on full embeddings
    embedding_columns = [f'embed_{i}' for i in range(128)]
    known_embeddings = df[embedding_columns].values
    input_embedding = input_embedding / np.linalg.norm(input_embedding)
    known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)

    distances = np.linalg.norm(known_embeddings - input_embedding, axis=1)
    df['distance'] = distances
    df_sorted = df.sort_values('distance')

    print(f"\nTop {top_k} closest images in the Database :\n")
    for idx, row in df_sorted.head(top_k).iterrows():
        print(f"Image: {row['image']} | Person: {row['person']} | Distance: {row['distance']:.5f}")

    # Check if same image present
    same_img_present = False
    print(f"\nTop matches with threshold (with normalized embeddings):\n")
    for idx, row in df_sorted.head(top_k).iterrows():
        # if row['distance'] < 0.05:
        # threshold=suggest_threshold(distances, percentile=95)
        # print("Threshold : ",threshold)
        if row['distance'] < 0.25:

            print(f"Image: {row['image']} | Person: {row['person']} | Distance: {row['distance']:.5f}")
            if row['distance'] == 0.0:
                same_img_present = True
                print("Same image present in the csv file.")
                break

    # Add image if not present
    if not same_img_present:
        print("")
        print("Image is not present in the database.")
        x = int(input("Do you want to add this image to the database? (1 for yes, 0 for no) : "))
        if x:
            df_unique_names = df['person'].unique()
            print("Unique names in the CSV file are:")
            for ele in df_unique_names:
                print(ele)
            person_name = input("Enter the name for the person in the image (Exact if exist already) : ")
            count_person_name = df[df['person'] == person_name].shape[0]
            if count_person_name > 50:
                print("This person already has 50 images in the database, no need to add more.")
            else:
                new_row = {
                    'image': os.path.basename(input_image_path),
                    'person': person_name,
                }
                for i in range(128):
                    new_row[f'embed_{i}'] = input_embedding[i]

                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"Added {os.path.basename(input_image_path)} to the CSV file.")

    return df_sorted.head(top_k)

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    input_image_path = file_path
    csv_path = "database.csv"  # Your embeddings CSV

    find_most_similar(input_image_path, csv_path, top_k=5)

main()