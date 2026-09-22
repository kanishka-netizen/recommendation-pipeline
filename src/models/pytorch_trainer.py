from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.trainer import ModelTrainer
from src.models.two_tower import TwoTowerRecommender


class PyTorchModelTrainer(ModelTrainer):
    """
    Trains and persists the two-tower recommendation model.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 0.001,
    ):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.user_to_index = {}
        self.item_to_index = {}
        self.user_positive_items = {}

    def train(self, data):
        """
        Train a two-tower model using positive interactions
        and sampled negative items.

        Expected columns:
            user_id
            item_id
            reward
            timestamp
        """

        if data.empty:
            raise ValueError("Training data is empty.")

        # Keep only positive interactions.
        positive_data = data[data["reward"] > 0].copy()

        if positive_data.empty:
            raise ValueError(
                "No positive interactions found in training data."
            )

        # Remove duplicate user-item interactions.
        positive_data = positive_data.drop_duplicates(
            subset=["user_id", "item_id"]
        )

        # Create mappings from original IDs to integer indices.
        users = sorted(positive_data["user_id"].unique())
        items = sorted(positive_data["item_id"].unique())

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(users)
        }

        self.item_to_index = {
            item_id: index
            for index, item_id in enumerate(items)
        }

        # Convert interactions into integer tensors.
        user_ids = torch.tensor(
            [
                self.user_to_index[user_id]
                for user_id in positive_data["user_id"]
            ],
            dtype=torch.long,
        )

        item_ids = torch.tensor(
            [
                self.item_to_index[item_id]
                for item_id in positive_data["item_id"]
            ],
            dtype=torch.long,
        )

        # Store the items each user has already interacted with.
        user_positive_items = {}

        for user, item in zip(user_ids.tolist(), item_ids.tolist()):
            if user not in user_positive_items:
                user_positive_items[user] = set()

            user_positive_items[user].add(item)

        self.user_positive_items = user_positive_items
        model = TwoTowerRecommender(
            num_users=len(self.user_to_index),
            num_items=len(self.item_to_index),
            embedding_dim=self.embedding_dim,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
        )

        model.train()

        dataset_size = len(user_ids)
        num_items = len(self.item_to_index)

        for epoch in range(self.epochs):

            total_loss = 0.0
            batch_count = 0

            # Shuffle the training examples.
            permutation = torch.randperm(dataset_size)

            shuffled_users = user_ids[permutation]
            shuffled_items = item_ids[permutation]

            for start in range(
                0,
                dataset_size,
                self.batch_size,
            ):

                end = min(
                    start + self.batch_size,
                    dataset_size,
                )

                batch_users = shuffled_users[start:end]
                batch_positive_items = shuffled_items[start:end]

                # Sample one negative item for every positive item.
                negative_items = torch.randint(
                    low=0,
                    high=num_items,
                    size=batch_positive_items.shape,
                )

                # Make sure the sampled negative item is not
                # something that user has already interacted with.
                invalid = torch.tensor(
                    [
                        negative_item.item()
                        in user_positive_items[user.item()]
                        for user, negative_item
                        in zip(batch_users, negative_items)
                    ],
                    dtype=torch.bool,
                )

                # Resample invalid negatives.
                while invalid.any():

                    count = invalid.sum().item()

                    negative_items[invalid] = torch.randint(
                        low=0,
                        high=num_items,
                        size=(count,),
                    )

                    invalid = torch.tensor(
                        [
                            negative_item.item()
                            in user_positive_items[user.item()]
                            for user, negative_item
                            in zip(batch_users, negative_items)
                        ],
                        dtype=torch.bool,
                    )

                # Calculate positive scores.
                positive_scores = model(
                    batch_users,
                    batch_positive_items,
                )

                # Calculate negative scores.
                negative_scores = model(
                    batch_users,
                    negative_items,
                )

                # BPR-style ranking loss.
                #
                # We want:
                #
                # positive_score > negative_score
                #
                loss = F.softplus(
                    negative_scores - positive_scores
                ).mean()

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            average_loss = total_loss / batch_count

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- Loss: {average_loss:.4f}"
            )

        return model

    def save(self, model, path: str | Path):
        """
        Save the model and ID mappings.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "user_to_index": self.user_to_index,
            "item_to_index": self.item_to_index,
            "embedding_dim": self.embedding_dim,
            "user_positive_items": self.user_positive_items,
        }

        torch.save(
            checkpoint,
            path,
        )

    def load(self, path: str | Path):
        """
        Load a previously trained model.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.user_to_index = checkpoint[
            "user_to_index"
        ]

        self.item_to_index = checkpoint[
            "item_to_index"
        ]

        self.user_positive_items = checkpoint[
            "user_positive_items"
        ]

        model = TwoTowerRecommender(
            num_users=len(self.user_to_index),
            num_items=len(self.item_to_index),
            embedding_dim=checkpoint[
                "embedding_dim"
            ],
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        return model
    def recommend(self, model, user_id, top_k=10):
        """
        Generate top-K recommendations for a user.

        Items the user has already interacted with are excluded.
        """

        if user_id not in self.user_to_index:
            raise ValueError(
                f"Unknown user ID: {user_id}"
            )

        user_index = self.user_to_index[user_id]

        user_tensor = torch.tensor(
            [user_index],
            dtype=torch.long,
        )

        item_indices = torch.arange(
            len(self.item_to_index),
            dtype=torch.long,
        )

        model.eval()

        with torch.no_grad():

            user_vector = model.encode_users(
                user_tensor
            )

            item_vectors = model.encode_items(
                item_indices
            )

            scores = user_vector @ item_vectors.T

        # Find items already interacted with by this user.
        #
        # item_to_index maps:
        # original item ID -> internal index
        #
        # We need the reverse mapping here.
        index_to_item = {
            index: item_id
            for item_id, index
            in self.item_to_index.items()
        }

        # Remove items the user has already interacted with.
        consumed_items = self.user_positive_items.get(
            user_index,
            set(),
        )

        for item_index in consumed_items:
            scores[0, item_index] = float("-inf")

        top_k = min(
            top_k,
            len(item_indices),
        )

        top_scores, top_indices = torch.topk(
            scores.squeeze(0),
            k=top_k,
        )

        recommendations = [
            (
                index_to_item[index.item()],
                score.item(),
            )
            for index, score
            in zip(top_indices, top_scores)
        ]

        return recommendations