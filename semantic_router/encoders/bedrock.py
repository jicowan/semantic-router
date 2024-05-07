"""
This module provides the BedrockEncoder class for generating embeddings using Amazon's Bedrock Platform.

The BedrockEncoder class is a subclass of BaseEncoder and utilizes the TextEmbeddingModel from the
Amazon's Bedrock Platform to generate embeddings for given documents. It requires an AWS Access Key ID
and AWS Secret Access Key and supports customization of the pre-trained model, score threshold, and region.

Example usage:

    from semantic_router.encoders.bedrock_encoder import BedrockEncoder

    encoder = BedrockEncoder(aws_access_key_id="your-access-key-id", aws_secret_access_key="your-secret-key", region="your-region")
    embeddings = encoder(["document1", "document2"])

Classes:
    BedrockEncoder: A class for generating embeddings using the Bedrock Platform.
"""
import json
import os
import boto3
from typing import Any, List, Optional

from semantic_router.encoders import BaseEncoder
from defaults import EncoderDefault


class BedrockEncoder(BaseEncoder):
    """BedrockEncoder class for generating embeddings using Amazon's Bedrock Platform.

    Attributes:
        client: An instance of the TextEmbeddingModel client.
        type: The type of the encoder, which is "bedrock".
    """

    client: Optional[Any] = None
    type: str = "bedrock"

    def __init__(
        self,
        model_id: Optional[str] = None,
        score_threshold: float = 0.75,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Initializes the BedrockEncoder.

        Args:
            model_id: The name of the pre-trained model to use for embedding.
                If not provided, the default model specified in EncoderDefault will
                be used.
            score_threshold: The threshold for similarity scores.
            aws_access_key_id: The AWS access key id for an IAM principle.
                If not provided, it will be retrieved from the AWS_ACCESS_KEY_ID
                environment variable.
            aws_secret_access_key: The secret access key for an IAM principle.
                If not provided, it will be retrieved from the AWS_SECRET_KEY
                environment variable.
            region: The location of the Bedrock resources.
                If not provided, it will be retrieved from the AWS_REGION
                environment variable, defaulting to "us-west-2"

        Raises:
            ValueError: If the Bedrock Platform client fails to initialize.
        """
        if model_id is None:
            model_id = EncoderDefault.BEDROCK.value["embedding_model"]

        super().__init__(name=model_id, score_threshold=score_threshold)

        self.client = self._initialize_client(aws_access_key_id, aws_secret_access_key, region)

    def _initialize_client(self, aws_access_key_id, aws_secret_access_key, region):
        """Initializes the Bedrock client.

        Args:
            aws_access_key_id: The Amazon access key ID.
            aws_secret_access_key: The Amazon secret key.
            region: The location of the AI Platform resources.

        Returns:
            An instance of the TextEmbeddingModel client.

        Raises:
            ImportError: If the required Bedrock libraries are not
            installed.
            ValueError: If the Bedrock client fails to initialize.
        """
        try:
            from boto3 import client
        except ImportError:
            raise ImportError(
                "Please install Amazon's Boto3 client library to use the BedrockEncoder. "
                "You can install them with: "
                "`pip install boto3`"
            )

        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = region or os.getenv("AWS_REGION", "us-west-2")

        if aws_access_key_id is None:
            raise ValueError("AWS access key ID cannot be 'None'.")

        if aws_secret_key is None:
            raise ValueError("AWS secret access key cannot be 'None'.")

        try:
            bedrock_client = boto3.client('bedrock-runtime',
                                          aws_access_key_id=aws_access_key_id,
                                          aws_secret_access_key=aws_secret_access_key,
                                          region_name=region
                                          )
        except Exception as err:
            raise ValueError(
                f"The Bedrock client failed to initialize. Error: {err}"
            ) from err

        return bedrock_client

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Generates embeddings for the given documents.

        Args:
            docs: A list of strings representing the documents to embed.

        Returns:
            A list of lists, where each inner list contains the embedding values for a
            document.

        Raises:
            ValueError: If the Bedrock Platform client is not initialized or if the
            API call fails.
        """
        if self.client is None:
            raise ValueError("Bedrock client is not initialized.")

        responses = []
        for doc in docs:
            if self.name == "cohere.embed-english-v3":
                body = json.dumps(
                    {
                        "texts": [doc],
                        "input_type": "clustering"
                    }
                )
                accept = "*/*"
                embedding = self.client.invoke_model(body=body,
                                                      contentType="application/json",
                                                      accept=accept,
                                                      modelId=self.name
                                                      )
                r = json.loads(embedding.get('body').read())
                responses.append(r.get('embeddings'))
            else:
                body = json.dumps(
                    {
                        "inputText": doc,
                    }
                )
                try:
                    embedding = self.client.invoke_model(body=body,
                                                         contentType="application/json",
                                                         accept="application/json",
                                                         modelId="amazon.titan-embed-text-v1"
                                                         )
                    r = json.loads(embedding.get('body').read())
                    responses.append(r.get('embedding'))
                except Exception as e:
                    raise ValueError(f"Bedrock Platform API call failed. Error: {e}") from e
        return responses
