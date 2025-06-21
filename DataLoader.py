from datasets import load_dataset

class AmazonReviews:
    def __init__(self):
       self.dataset_name = "McAuley-Lab/Amazon-Reviews-2023"
        
    def get_raw_data(self, category: str):
        self.dataset = load_dataset(self.dataset_name, f"raw_review_{category}", trust_remote_code=True)
        return self.dataset

    def get_meta_data(self, category: str):
        self.meta_data = load_dataset(self.dataset_name, f"raw_meta_{category}", split="full", trust_remote_code=True)
        return self.meta_data

if __name__ == "__main__":
    amazon_reviews = AmazonReviews()
    print(amazon_reviews.get_meta_data("Movies_and_TV"))





