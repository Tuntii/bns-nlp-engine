"""
CLI interface for bns-nlp-engine.

This module provides command-line interface for Turkish NLP operations
including preprocessing, embedding, search, and classification.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from bnsnlp.core.config import Config
from bnsnlp.core.exceptions import AdapterError
from bnsnlp.preprocess.turkish import TurkishPreprocessor

app = typer.Typer(
    name="bnsnlp",
    help="Turkish NLP Engine - Command-line interface for text processing",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from bnsnlp.__version__ import __version__
        typer.echo(f"bns-nlp-engine version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
) -> None:
    """
    Turkish NLP Engine CLI.
    
    Process Turkish text with preprocessing, embeddings, search, and classification.
    """
    pass


@app.command()
def preprocess(
    input_text: Annotated[
        Optional[str],
        typer.Option(
            "--input",
            "-i",
            help="Input text or file path. If not provided, reads from stdin.",
        ),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If not provided, prints to stdout.",
            dir_okay=False,
        ),
    ] = None,
    lowercase: Annotated[
        bool,
        typer.Option(
            "--lowercase/--no-lowercase",
            help="Convert text to lowercase.",
        ),
    ] = True,
    remove_punctuation: Annotated[
        bool,
        typer.Option(
            "--remove-punctuation/--keep-punctuation",
            help="Remove punctuation marks.",
        ),
    ] = True,
    remove_stopwords: Annotated[
        bool,
        typer.Option(
            "--remove-stopwords/--keep-stopwords",
            help="Remove Turkish stop words.",
        ),
    ] = True,
    lemmatize: Annotated[
        bool,
        typer.Option(
            "--lemmatize/--no-lemmatize",
            help="Apply lemmatization.",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
        ),
    ] = False,
) -> None:
    """
    Preprocess Turkish text.
    
    Applies normalization, tokenization, stop word removal, and lemmatization
    to Turkish text. Input can be provided as a string, file path, or from stdin.
    
    Examples:
    
        # Process text from stdin
        echo "Merhaba DÜNYA!" | bnsnlp preprocess
        
        # Process text from file
        bnsnlp preprocess -i input.txt -o output.json
        
        # Process with custom options
        bnsnlp preprocess -i text.txt --no-lemmatize --keep-stopwords
    """
    try:
        # Load configuration
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        # Override config with CLI options
        preprocess_config = {
            'lowercase': lowercase,
            'remove_punctuation': remove_punctuation,
            'remove_stopwords': remove_stopwords,
            'lemmatize': lemmatize,
        }
        
        # Get input text
        if input_text:
            # Check if it's a file path
            input_path = Path(input_text)
            if input_path.exists() and input_path.is_file():
                text = input_path.read_text(encoding='utf-8')
                if verbose:
                    console.print(f"[blue]Reading from file: {input_text}[/blue]")
            else:
                # Treat as direct text input
                text = input_text
        else:
            # Read from stdin
            if verbose:
                console.print("[blue]Reading from stdin...[/blue]")
            text = sys.stdin.read()
        
        if not text.strip():
            console.print("[red]Error: No input text provided[/red]", err=True)
            raise typer.Exit(1)
        
        # Initialize preprocessor
        preprocessor = TurkishPreprocessor(preprocess_config)
        
        if verbose:
            console.print("[blue]Processing text...[/blue]")
        
        # Process text
        result = asyncio.run(preprocessor.process(text))
        
        # Prepare output
        output_data = {
            'text': result.text,
            'tokens': result.tokens,
            'metadata': result.metadata,
        }
        
        output_json = json.dumps(output_data, ensure_ascii=False, indent=2)
        
        # Write output
        if output_file:
            output_file.write_text(output_json, encoding='utf-8')
            if verbose:
                console.print(f"[green]Output written to: {output_file}[/green]")
        else:
            console.print(output_json)
        
        if verbose:
            console.print(
                f"[green]✓ Processed {result.metadata['token_count']} tokens[/green]"
            )
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]", err=True)
        if verbose:
            import traceback
            console.print(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@app.command()
def embed(
    input_text: Annotated[
        Optional[str],
        typer.Option(
            "--input",
            "-i",
            help="Input text or file path. If not provided, reads from stdin.",
        ),
    ] = None,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Embedding provider (openai, cohere, huggingface).",
        ),
    ] = "openai",
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name for embeddings.",
        ),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If not provided, prints to stdout.",
            dir_okay=False,
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            help="API key for the embedding provider. Can also be set via environment variable.",
            envvar=f"BNSNLP_EMBED_API_KEY",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
        ),
    ] = False,
) -> None:
    """
    Generate embeddings for text.
    
    Converts text into vector embeddings using the specified provider.
    Supports OpenAI, Cohere, and HuggingFace models.
    
    Examples:
    
        # Embed text using OpenAI
        echo "Merhaba dünya" | bnsnlp embed --provider openai
        
        # Embed text from file using Cohere
        bnsnlp embed -i input.txt -p cohere -o embeddings.json
        
        # Use custom model
        bnsnlp embed -i text.txt -p openai -m text-embedding-3-large
    """
    try:
        # Load configuration
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        # Get input text
        if input_text:
            # Check if it's a file path
            input_path = Path(input_text)
            if input_path.exists() and input_path.is_file():
                text = input_path.read_text(encoding='utf-8')
                if verbose:
                    console.print(f"[blue]Reading from file: {input_text}[/blue]")
            else:
                # Treat as direct text input
                text = input_text
        else:
            # Read from stdin
            if verbose:
                console.print("[blue]Reading from stdin...[/blue]")
            text = sys.stdin.read()
        
        if not text.strip():
            console.print("[red]Error: No input text provided[/red]", err=True)
            raise typer.Exit(1)
        
        # Prepare embedder configuration
        embed_config = {
            'provider': provider,
            'api_key': api_key,
        }
        
        if model:
            embed_config['model'] = model
        else:
            embed_config['model'] = config.embed.model
        
        # Import the appropriate embedder
        if verbose:
            console.print(f"[blue]Using {provider} embedder...[/blue]")
        
        embedder = None
        if provider == 'openai':
            from bnsnlp.embed.openai import OpenAIEmbedder
            embedder = OpenAIEmbedder(embed_config)
        elif provider == 'cohere':
            from bnsnlp.embed.cohere import CohereEmbedder
            embedder = CohereEmbedder(embed_config)
        elif provider == 'huggingface':
            from bnsnlp.embed.huggingface import HuggingFaceEmbedder
            embedder = HuggingFaceEmbedder(embed_config)
        else:
            console.print(
                f"[red]Error: Unknown provider '{provider}'. "
                f"Must be one of: openai, cohere, huggingface[/red]",
                err=True
            )
            raise typer.Exit(1)
        
        if verbose:
            console.print("[blue]Generating embeddings...[/blue]")
        
        # Generate embeddings
        result = asyncio.run(embedder.embed(text))
        
        # Prepare output
        output_data = {
            'embeddings': result.embeddings,
            'model': result.model,
            'dimensions': result.dimensions,
            'metadata': result.metadata,
        }
        
        output_json = json.dumps(output_data, ensure_ascii=False, indent=2)
        
        # Write output
        if output_file:
            output_file.write_text(output_json, encoding='utf-8')
            if verbose:
                console.print(f"[green]Output written to: {output_file}[/green]")
        else:
            console.print(output_json)
        
        if verbose:
            console.print(
                f"[green]✓ Generated {len(result.embeddings)} embedding(s) "
                f"with {result.dimensions} dimensions[/green]"
            )
    
    except AdapterError as e:
        console.print(f"[red]Adapter Error: {str(e)}[/red]", err=True)
        if verbose:
            console.print(f"[yellow]Context: {e.context}[/yellow]", err=True)
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]", err=True)
        if verbose:
            import traceback
            console.print(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(help="Search query text."),
    ],
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Search backend provider (faiss, qdrant, pinecone).",
        ),
    ] = "faiss",
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of top results to return.",
        ),
    ] = 10,
    embed_provider: Annotated[
        str,
        typer.Option(
            "--embed-provider",
            help="Embedding provider for query (openai, cohere, huggingface).",
        ),
    ] = "openai",
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If not provided, prints to stdout.",
            dir_okay=False,
        ),
    ] = None,
    filters: Annotated[
        Optional[str],
        typer.Option(
            "--filters",
            help="JSON string of filters to apply to search.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
        ),
    ] = False,
) -> None:
    """
    Search for similar documents.
    
    Performs semantic search using vector similarity. First embeds the query,
    then searches for similar documents in the specified backend.
    
    Examples:
    
        # Search using FAISS
        bnsnlp search "Türkçe NLP" --provider faiss --top-k 5
        
        # Search with filters
        bnsnlp search "machine learning" --filters '{"category": "tech"}'
        
        # Search using Qdrant
        bnsnlp search "doğal dil işleme" -p qdrant -k 10
    """
    try:
        # Load configuration
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        # Parse filters if provided
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError as e:
                console.print(
                    f"[red]Error: Invalid JSON in filters: {str(e)}[/red]",
                    err=True
                )
                raise typer.Exit(1)
        
        # Step 1: Embed the query
        if verbose:
            console.print(f"[blue]Embedding query using {embed_provider}...[/blue]")
        
        embed_config = {
            'provider': embed_provider,
            'model': config.embed.model,
            'api_key': config.embed.api_key,
        }
        
        embedder = None
        if embed_provider == 'openai':
            from bnsnlp.embed.openai import OpenAIEmbedder
            embedder = OpenAIEmbedder(embed_config)
        elif embed_provider == 'cohere':
            from bnsnlp.embed.cohere import CohereEmbedder
            embedder = CohereEmbedder(embed_config)
        elif embed_provider == 'huggingface':
            from bnsnlp.embed.huggingface import HuggingFaceEmbedder
            embedder = HuggingFaceEmbedder(embed_config)
        else:
            console.print(
                f"[red]Error: Unknown embed provider '{embed_provider}'. "
                f"Must be one of: openai, cohere, huggingface[/red]",
                err=True
            )
            raise typer.Exit(1)
        
        embed_result = asyncio.run(embedder.embed(query))
        query_embedding = embed_result.embeddings[0]
        
        if verbose:
            console.print(
                f"[blue]Query embedded with {embed_result.dimensions} dimensions[/blue]"
            )
        
        # Step 2: Search
        if verbose:
            console.print(f"[blue]Searching using {provider}...[/blue]")
        
        search_config = {
            'provider': provider,
            'top_k': top_k,
        }
        
        searcher = None
        if provider == 'faiss':
            from bnsnlp.search.faiss import FAISSSearch
            searcher = FAISSSearch(search_config)
        elif provider == 'qdrant':
            from bnsnlp.search.qdrant import QdrantSearch
            searcher = QdrantSearch(search_config)
        elif provider == 'pinecone':
            from bnsnlp.search.pinecone import PineconeSearch
            searcher = PineconeSearch(search_config)
        else:
            console.print(
                f"[red]Error: Unknown search provider '{provider}'. "
                f"Must be one of: faiss, qdrant, pinecone[/red]",
                err=True
            )
            raise typer.Exit(1)
        
        search_result = asyncio.run(
            searcher.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filter_dict
            )
        )
        
        # Prepare output
        output_data = {
            'query': query,
            'results': [
                {
                    'id': r.id,
                    'score': r.score,
                    'text': r.text,
                    'metadata': r.metadata,
                }
                for r in search_result.results
            ],
            'query_time_ms': search_result.query_time_ms,
            'metadata': search_result.metadata,
        }
        
        output_json = json.dumps(output_data, ensure_ascii=False, indent=2)
        
        # Write output
        if output_file:
            output_file.write_text(output_json, encoding='utf-8')
            if verbose:
                console.print(f"[green]Output written to: {output_file}[/green]")
        else:
            console.print(output_json)
        
        if verbose:
            console.print(
                f"[green]✓ Found {len(search_result.results)} result(s) "
                f"in {search_result.query_time_ms:.2f}ms[/green]"
            )
    
    except AdapterError as e:
        console.print(f"[red]Adapter Error: {str(e)}[/red]", err=True)
        if verbose:
            console.print(f"[yellow]Context: {e.context}[/yellow]", err=True)
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]", err=True)
        if verbose:
            import traceback
            console.print(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@app.command()
def classify(
    input_text: Annotated[
        Optional[str],
        typer.Option(
            "--input",
            "-i",
            help="Input text or file path. If not provided, reads from stdin.",
        ),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If not provided, prints to stdout.",
            dir_okay=False,
        ),
    ] = None,
    intent_model: Annotated[
        Optional[str],
        typer.Option(
            "--intent-model",
            help="Intent classification model name.",
        ),
    ] = None,
    entity_model: Annotated[
        Optional[str],
        typer.Option(
            "--entity-model",
            help="Entity recognition model name.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
        ),
    ] = False,
) -> None:
    """
    Classify intent and extract entities from text.
    
    Performs intent classification and named entity recognition on Turkish text.
    Returns structured results with confidence scores.
    
    Examples:
    
        # Classify text from stdin
        echo "Yarın hava nasıl olacak?" | bnsnlp classify
        
        # Classify text from file
        bnsnlp classify -i input.txt -o results.json
        
        # Use custom models
        bnsnlp classify -i text.txt --intent-model custom-intent --entity-model custom-ner
    """
    try:
        # Load configuration
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        # Get input text
        if input_text:
            # Check if it's a file path
            input_path = Path(input_text)
            if input_path.exists() and input_path.is_file():
                text = input_path.read_text(encoding='utf-8')
                if verbose:
                    console.print(f"[blue]Reading from file: {input_text}[/blue]")
            else:
                # Treat as direct text input
                text = input_text
        else:
            # Read from stdin
            if verbose:
                console.print("[blue]Reading from stdin...[/blue]")
            text = sys.stdin.read()
        
        if not text.strip():
            console.print("[red]Error: No input text provided[/red]", err=True)
            raise typer.Exit(1)
        
        # Prepare classifier configuration
        classify_config = {}
        
        if intent_model:
            classify_config['intent_model'] = intent_model
        
        if entity_model:
            classify_config['entity_model'] = entity_model
        
        # Initialize classifier
        if verbose:
            console.print("[blue]Initializing Turkish classifier...[/blue]")
        
        from bnsnlp.classify.turkish import TurkishClassifier
        classifier = TurkishClassifier(classify_config)
        
        if verbose:
            console.print("[blue]Classifying text...[/blue]")
        
        # Classify text
        result = asyncio.run(classifier.classify(text))
        
        # Prepare output
        output_data = {
            'intent': result.intent,
            'intent_confidence': result.intent_confidence,
            'entities': [
                {
                    'text': e.text,
                    'type': e.type,
                    'start': e.start,
                    'end': e.end,
                    'confidence': e.confidence,
                }
                for e in result.entities
            ],
            'metadata': result.metadata,
        }
        
        output_json = json.dumps(output_data, ensure_ascii=False, indent=2)
        
        # Write output
        if output_file:
            output_file.write_text(output_json, encoding='utf-8')
            if verbose:
                console.print(f"[green]Output written to: {output_file}[/green]")
        else:
            console.print(output_json)
        
        if verbose:
            console.print(
                f"[green]✓ Intent: {result.intent} "
                f"(confidence: {result.intent_confidence:.2f})[/green]"
            )
            console.print(
                f"[green]✓ Found {len(result.entities)} entit(y/ies)[/green]"
            )
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]", err=True)
        if verbose:
            import traceback
            console.print(traceback.format_exc(), err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
