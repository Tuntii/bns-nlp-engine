"""
Tests for classification module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bnsnlp.classify import BaseClassifier, ClassifyResult, Entity, TurkishClassifier
from bnsnlp.core.exceptions import ProcessingError


class TestEntity:
    """Tests for Entity model."""

    def test_entity_creation(self):
        """Test creating an Entity."""
        entity = Entity(
            text="Istanbul",
            type="LOCATION",
            start=0,
            end=8,
            confidence=0.95
        )

        assert entity.text == "Istanbul"
        assert entity.type == "LOCATION"
        assert entity.start == 0
        assert entity.end == 8
        assert entity.confidence == 0.95

    def test_entity_confidence_validation(self):
        """Test that confidence is validated to be between 0 and 1."""
        # Valid confidence
        entity = Entity(text="test", type="TEST", start=0, end=4, confidence=0.5)
        assert entity.confidence == 0.5

        # Invalid confidence > 1
        with pytest.raises(Exception):
            Entity(text="test", type="TEST", start=0, end=4, confidence=1.5)

        # Invalid confidence < 0
        with pytest.raises(Exception):
            Entity(text="test", type="TEST", start=0, end=4, confidence=-0.1)


class TestClassifyResult:
    """Tests for ClassifyResult model."""

    def test_classify_result_creation(self):
        """Test creating a ClassifyResult."""
        entities = [
            Entity(text="Istanbul", type="LOCATION", start=0, end=8, confidence=0.95)
        ]
        result = ClassifyResult(
            intent="travel_query",
            intent_confidence=0.92,
            entities=entities
        )

        assert result.intent == "travel_query"
        assert result.intent_confidence == 0.92
        assert len(result.entities) == 1
        assert result.entities[0].text == "Istanbul"
        assert isinstance(result.metadata, dict)

    def test_classify_result_with_metadata(self):
        """Test ClassifyResult with custom metadata."""
        metadata = {"model": "test-model", "device": "cpu"}
        result = ClassifyResult(
            intent="test_intent",
            intent_confidence=0.8,
            entities=[],
            metadata=metadata
        )

        assert result.metadata["model"] == "test-model"
        assert result.metadata["device"] == "cpu"

    def test_classify_result_empty_entities(self):
        """Test ClassifyResult with no entities."""
        result = ClassifyResult(
            intent="greeting",
            intent_confidence=0.99,
            entities=[]
        )

        assert result.intent == "greeting"
        assert len(result.entities) == 0

    def test_classify_result_intent_confidence_validation(self):
        """Test that intent_confidence is validated."""
        # Valid confidence
        result = ClassifyResult(intent="test", intent_confidence=0.5, entities=[])
        assert result.intent_confidence == 0.5

        # Invalid confidence > 1
        with pytest.raises(Exception):
            ClassifyResult(intent="test", intent_confidence=1.5, entities=[])

        # Invalid confidence < 0
        with pytest.raises(Exception):
            ClassifyResult(intent="test", intent_confidence=-0.1, entities=[])


class TestTurkishClassifier:
    """Tests for TurkishClassifier."""

    @pytest.fixture
    def mock_transformers(self):
        """Mock the transformers module."""
        mock_transformers = MagicMock()
        mock_torch = MagicMock()

        # Mock torch
        mock_torch.cuda.is_available.return_value = True
        
        # Mock pipeline function
        def mock_pipeline(task, model, device, **kwargs):
            mock_pipe = MagicMock()
            if task == 'text-classification':
                # Intent classification mock
                mock_pipe.return_value = [
                    {'label': 'positive', 'score': 0.92}
                ]
            elif task == 'ner':
                # NER mock
                mock_pipe.return_value = [
                    {
                        'word': 'Istanbul',
                        'entity_group': 'LOCATION',
                        'start': 0,
                        'end': 8,
                        'score': 0.95
                    }
                ]
            return mock_pipe
        
        mock_transformers.pipeline = mock_pipeline

        with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
            yield mock_transformers, mock_torch

    @pytest.fixture
    def turkish_config(self):
        """Create a test configuration for Turkish classifier."""
        return {
            "intent_model": "savasy/bert-base-turkish-sentiment-cased",
            "entity_model": "savasy/bert-turkish-ner-cased",
            "use_gpu": True,
            "batch_size": 8
        }

    def test_turkish_classifier_initialization(self, turkish_config, mock_transformers):
        """Test TurkishClassifier initialization."""
        classifier = TurkishClassifier(turkish_config)

        assert classifier.intent_model_name == "savasy/bert-base-turkish-sentiment-cased"
        assert classifier.entity_model_name == "savasy/bert-turkish-ner-cased"
        assert classifier.use_gpu is True
        assert classifier.batch_size == 8

    def test_turkish_classifier_default_config(self, mock_transformers):
        """Test TurkishClassifier with default configuration."""
        classifier = TurkishClassifier({})

        assert classifier.intent_model_name == "savasy/bert-base-turkish-sentiment-cased"
        assert classifier.entity_model_name == "savasy/bert-turkish-ner-cased"
        assert classifier.use_gpu is True
        assert classifier.batch_size == 8

    def test_turkish_classifier_custom_config(self, mock_transformers):
        """Test TurkishClassifier with custom configuration."""
        config = {
            "intent_model": "custom-intent-model",
            "entity_model": "custom-ner-model",
            "use_gpu": False,
            "batch_size": 16
        }

        classifier = TurkishClassifier(config)

        assert classifier.intent_model_name == "custom-intent-model"
        assert classifier.entity_model_name == "custom-ner-model"
        assert classifier.use_gpu is False
        assert classifier.batch_size == 16

    @pytest.mark.asyncio
    async def test_classify_single_text(self, turkish_config, mock_transformers):
        """Test classifying a single text."""
        mock_tf, mock_torch = mock_transformers
        
        # Setup mock pipelines
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.return_value = [{'label': 'positive', 'score': 0.92}]
        
        mock_entity_pipe = MagicMock()
        mock_entity_pipe.return_value = [
            {
                'word': 'Istanbul',
                'entity_group': 'LOCATION',
                'start': 0,
                'end': 8,
                'score': 0.95
            }
        ]
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            elif task == 'ner':
                return mock_entity_pipe
        
        mock_tf.pipeline = mock_pipeline

        classifier = TurkishClassifier(turkish_config)
        result = await classifier.classify("Istanbul'a gitmek istiyorum")

        assert isinstance(result, ClassifyResult)
        assert result.intent == "positive"
        assert result.intent_confidence == 0.92
        assert len(result.entities) == 1
        assert result.entities[0].text == "Istanbul"
        assert result.entities[0].type == "LOCATION"
        assert result.entities[0].confidence == 0.95
        assert result.metadata["intent_model"] == "savasy/bert-base-turkish-sentiment-cased"
        assert result.metadata["entity_model"] == "savasy/bert-turkish-ner-cased"

    @pytest.mark.asyncio
    async def test_classify_text_no_entities(self, turkish_config, mock_transformers):
        """Test classifying text with no entities."""
        mock_tf, mock_torch = mock_transformers
        
        # Setup mock pipelines
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.return_value = [{'label': 'neutral', 'score': 0.88}]
        
        mock_entity_pipe = MagicMock()
        mock_entity_pipe.return_value = []  # No entities
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            elif task == 'ner':
                return mock_entity_pipe
        
        mock_tf.pipeline = mock_pipeline

        classifier = TurkishClassifier(turkish_config)
        result = await classifier.classify("Merhaba")

        assert isinstance(result, ClassifyResult)
        assert result.intent == "neutral"
        assert result.intent_confidence == 0.88
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_classify_batch(self, turkish_config, mock_transformers):
        """Test classifying multiple texts."""
        mock_tf, mock_torch = mock_transformers
        
        # Setup mock pipelines for batch
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.return_value = [
            {'label': 'positive', 'score': 0.92},
            {'label': 'negative', 'score': 0.85}
        ]
        
        mock_entity_pipe = MagicMock()
        mock_entity_pipe.side_effect = [
            [{'word': 'Istanbul', 'entity_group': 'LOCATION', 'start': 0, 'end': 8, 'score': 0.95}],
            [{'word': 'Ankara', 'entity_group': 'LOCATION', 'start': 0, 'end': 6, 'score': 0.93}]
        ]
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            elif task == 'ner':
                return mock_entity_pipe
        
        mock_tf.pipeline = mock_pipeline

        classifier = TurkishClassifier(turkish_config)
        texts = ["Istanbul'a gitmek istiyorum", "Ankara'yı sevmiyorum"]
        results = await classifier.classify(texts)

        assert isinstance(results, list)
        assert len(results) == 2
        
        # First result
        assert results[0].intent == "positive"
        assert results[0].intent_confidence == 0.92
        assert len(results[0].entities) == 1
        assert results[0].entities[0].text == "Istanbul"
        
        # Second result
        assert results[1].intent == "negative"
        assert results[1].intent_confidence == 0.85
        assert len(results[1].entities) == 1
        assert results[1].entities[0].text == "Ankara"

    @pytest.mark.asyncio
    async def test_classify_gpu_detection(self, mock_transformers):
        """Test GPU detection in metadata."""
        mock_tf, mock_torch = mock_transformers
        
        # Test with GPU available
        mock_torch.cuda.is_available.return_value = True
        
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.return_value = [{'label': 'positive', 'score': 0.9}]
        
        mock_entity_pipe = MagicMock()
        mock_entity_pipe.return_value = []
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            elif task == 'ner':
                return mock_entity_pipe
        
        mock_tf.pipeline = mock_pipeline

        config = {"use_gpu": True}
        classifier = TurkishClassifier(config)
        result = await classifier.classify("test")

        assert result.metadata["device"] == "gpu"

    @pytest.mark.asyncio
    async def test_classify_cpu_fallback(self):
        """Test that classifier falls back to CPU when GPU is not available."""
        mock_transformers = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.return_value = [{'label': 'positive', 'score': 0.9}]
        
        mock_entity_pipe = MagicMock()
        mock_entity_pipe.return_value = []
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            elif task == 'ner':
                return mock_entity_pipe
        
        mock_transformers.pipeline = mock_pipeline

        with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
            config = {"use_gpu": True}
            classifier = TurkishClassifier(config)
            result = await classifier.classify("test")

            assert result.metadata["device"] == "cpu"

    @pytest.mark.asyncio
    async def test_classify_missing_transformers(self):
        """Test error when transformers is not installed."""
        # Don't mock transformers to simulate missing import
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            classifier = TurkishClassifier({})
            
            with pytest.raises(ProcessingError) as exc_info:
                await classifier.classify("test")
            
            assert "transformers and torch are required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_classify_model_loading_error(self):
        """Test error handling when model fails to load."""
        mock_transformers = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        # Mock pipeline to raise error
        def mock_pipeline(task, model, device, **kwargs):
            raise Exception("Model not found")
        
        mock_transformers.pipeline = mock_pipeline

        with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
            classifier = TurkishClassifier({})
            
            with pytest.raises(ProcessingError) as exc_info:
                await classifier.classify("test")
            
            assert "Failed to load" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_classify_inference_error(self, turkish_config, mock_transformers):
        """Test error handling during inference."""
        mock_tf, mock_torch = mock_transformers
        
        # Setup mock to raise error during inference
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.side_effect = Exception("Inference error")
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            return MagicMock()
        
        mock_tf.pipeline = mock_pipeline

        classifier = TurkishClassifier(turkish_config)
        
        with pytest.raises(ProcessingError) as exc_info:
            await classifier.classify("test")
        
        assert "Classification failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_classify_batch_inference_error(self, turkish_config, mock_transformers):
        """Test error handling during batch inference."""
        mock_tf, mock_torch = mock_transformers
        
        # Setup mock to raise error during batch inference
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.side_effect = Exception("Batch inference error")
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            return MagicMock()
        
        mock_tf.pipeline = mock_pipeline

        classifier = TurkishClassifier(turkish_config)
        
        with pytest.raises(ProcessingError) as exc_info:
            await classifier.classify(["test1", "test2"])
        
        assert "Batch classification failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_classify_multiple_entities(self, turkish_config, mock_transformers):
        """Test classifying text with multiple entities."""
        mock_tf, mock_torch = mock_transformers
        
        # Setup mock pipelines
        mock_intent_pipe = MagicMock()
        mock_intent_pipe.return_value = [{'label': 'travel', 'score': 0.94}]
        
        mock_entity_pipe = MagicMock()
        mock_entity_pipe.return_value = [
            {'word': 'Istanbul', 'entity_group': 'LOCATION', 'start': 0, 'end': 8, 'score': 0.95},
            {'word': 'Ahmet', 'entity_group': 'PERSON', 'start': 12, 'end': 17, 'score': 0.89},
            {'word': 'Pazartesi', 'entity_group': 'DATE', 'start': 21, 'end': 30, 'score': 0.92}
        ]
        
        def mock_pipeline(task, model, device, **kwargs):
            if task == 'text-classification':
                return mock_intent_pipe
            elif task == 'ner':
                return mock_entity_pipe
        
        mock_tf.pipeline = mock_pipeline

        classifier = TurkishClassifier(turkish_config)
        result = await classifier.classify("Istanbul'da Ahmet ile Pazartesi buluşacağım")

        assert isinstance(result, ClassifyResult)
        assert len(result.entities) == 3
        assert result.entities[0].type == "LOCATION"
        assert result.entities[1].type == "PERSON"
        assert result.entities[2].type == "DATE"
