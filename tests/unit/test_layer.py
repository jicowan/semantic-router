import pytest

from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder
from semantic_router.layer import RouteLayer
from semantic_router.route import Route


def mock_encoder_call(utterances):
    # Define a mapping of utterances to return values
    mock_responses = {
        "Hello": [0.1, 0.2, 0.3],
        "Hi": [0.4, 0.5, 0.6],
        "Goodbye": [0.7, 0.8, 0.9],
        "Bye": [1.0, 1.1, 1.2],
        "Au revoir": [1.3, 1.4, 1.5],
    }
    return [mock_responses.get(u, [0, 0, 0]) for u in utterances]


@pytest.fixture
def base_encoder():
    return BaseEncoder(name="test-encoder")


@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, "__call__", side_effect=mock_encoder_call)
    return CohereEncoder(name="test-cohere-encoder", cohere_api_key="test_api_key")


@pytest.fixture
def openai_encoder(mocker):
    mocker.patch.object(OpenAIEncoder, "__call__", side_effect=mock_encoder_call)
    return OpenAIEncoder(name="test-openai-encoder", openai_api_key="test_api_key")


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=["Hello", "Hi"]),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


class TestRouteLayer:
    def test_initialization(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        assert route_layer.score_threshold == 0.82
        assert len(route_layer.index) if route_layer.index is not None else 0 == 5
        assert (
            len(set(route_layer.categories))
            if route_layer.categories is not None
            else 0 == 2
        )

    def test_initialization_different_encoders(self, cohere_encoder, openai_encoder):
        route_layer_cohere = RouteLayer(encoder=cohere_encoder)
        assert route_layer_cohere.score_threshold == 0.3

        route_layer_openai = RouteLayer(encoder=openai_encoder)
        assert route_layer_openai.score_threshold == 0.82

    def test_add_route(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        route1 = Route(name="Route 1", utterances=["Yes", "No"])
        route2 = Route(name="Route 2", utterances=["Maybe", "Sure"])

        route_layer.add(route=route1)
        assert route_layer.index is not None and route_layer.categories is not None
        assert len(route_layer.index) == 2
        assert len(set(route_layer.categories)) == 1
        assert set(route_layer.categories) == {"Route 1"}

        route_layer.add(route=route2)
        assert len(route_layer.index) == 4
        assert len(set(route_layer.categories)) == 2
        assert set(route_layer.categories) == {"Route 1", "Route 2"}

    def test_add_multiple_routes(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder)
        route_layer._add_routes(routes=routes)
        assert route_layer.index is not None and route_layer.categories is not None
        assert len(route_layer.index) == 5
        assert len(set(route_layer.categories)) == 2

    def test_query_and_classification(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        query_result = route_layer("Hello")
        assert query_result in ["Route 1", "Route 2"]

    def test_query_with_no_index(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        assert route_layer("Anything") is None

    def test_semantic_classify(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9]

    def test_semantic_classify_multiple_routes(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
                {"route": "Route 1", "score": 0.8},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9, 0.8]

    def test_pass_threshold(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        assert not route_layer._pass_threshold([], 0.5)
        assert route_layer._pass_threshold([0.6, 0.7], 0.5)

    def test_failover_score_threshold(self, base_encoder):
        route_layer = RouteLayer(encoder=base_encoder)
        assert route_layer.score_threshold == 0.82


# Add more tests for edge cases and error handling as needed.