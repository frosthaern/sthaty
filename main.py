import argparse

from modules.classes.CodeBertTokenizerEncode import CodeBertTokenizeEncode
from modules.classes.Jsonl import Jsonl
from modules.classes.AttentionExtractor import AttentionExtractor
from modules.classes.AttentionAnalyzer import AttentionAnalyzer


def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Analyze code attention patterns.")
  parser.add_argument("-n", type=int, default=100, help="Number of code points to process")
  parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size for processing (default: 32)"
  )
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to run on (e.g., 'cuda', 'cpu'). Auto-detects if not specified.",
  )
  args = parser.parse_args()

  print(f"Processing {args.n} code points with batch size {args.batch_size}")

  try:
    # Initialize data pipeline
    print("Loading data...")
    data_loader = Jsonl("dataset.jsonl", max_lines=args.n, batch_size=args.batch_size)

    print("Tokenizing and encoding...")
    encoder = CodeBertTokenizeEncode(data_loader, batch_size=args.batch_size, show_progress=True)

    print("Extracting attention...")
    attention_extractor = AttentionExtractor(
      encoder, batch_size=args.batch_size, device=args.device, show_progress=True
    )

    print("Analyzing attention patterns...")
    analyzer = AttentionAnalyzer(attention_extractor, show_progress=True)

    # Process entropy
    print("Calculating entropy...")
    entropy_values = []
    for entropy in analyzer.entropy():
      entropy_values.append(entropy.flatten())

    # Process UMAP
    print("Running UMAP...")
    umap_results = []
    for result in analyzer.umap(entropy_values):
      umap_results.append(result)
      print(f"Processed UMAP result shape: {result.shape}")

    print("\nProcessing complete!")
    print(f"Total entropy values processed: {len(entropy_values)}")
    print(f"Total UMAP results generated: {len(umap_results)}")

  except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise


if __name__ == "__main__":
  main()
