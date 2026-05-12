from paper import get_papers, save_papers, get_other_papers
from reviewer.review import PeerReviewSystem
from utils.print_utils import timer
import os
import argparse


@timer
def main():
    parser = argparse.ArgumentParser(description="Peer Review System with configurable parameters.")
    parser.add_argument(
        '--venue',
        type=str,
        default='iclr_2025',
        help='Venue. Default: iclr_2025'
    )
    parser.add_argument(
        '--rank_method',
        type=str,
        default='bradley_terry',
        help='Rank method. Default: bradley_terry'
    )
    parser.add_argument(
        '--target-percentage',
        type=float,
        default=0.05,
        help='Target percentage for Bradley-Terry ranking. Default: 0.05'
    )
    parser.add_argument(
        '--keep-number',
        type=int,
        default=1e8,
        help='Number of papers to keep during testing. Default: 1e8'
    )
    parser.add_argument(
        '--comparison-model',
        type=str,
        required=True,
        help='Model name used to generate comparison file name, e.g., Qwen2.5-7B-Instruct'
    )
    parser.add_argument(
        '--keep-random-seed',
        type=int,
        default=42,
        help='Random seed for keeping papers. Default: 42'
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='If set, generate paper pairs and comparisons. Default: False'
    )
    parser.add_argument(
        '--rank',
        action='store_true',
        help='If set, execute Bardley-Terry ranking. Default: False'
    )
    parser.add_argument(
        '--no-random',
        action='store_true',
        help='If set, do not use random comparison. Default: False'
    )
    parser.add_argument(
        '--no-similarity',
        action='store_true',
        help='If set, do not use similarity comparison. Default: False'
    )
    args = parser.parse_args()

    # Construct Paths
    venue = args.venue
    model_name = os.path.basename(args.comparison_model)
    paper_path = f"./data/{venue}/papers.csv"
    distribution_path = f"./data/{venue}/distribution/distribution.csv"
    pair_types = {
        (False, False, False): ["standard", "random"],
        (True, False, False): ["standard"],
        (False, True, False): ["random"],
    }[(args.no_random, args.no_similarity)]
    comparison_paths = [
        f"./data/{venue}/comparisons-{pair_type}_{model_name}.csv" for pair_type in pair_types
    ]
    score_path = f"./data/{venue}/scores_{args.target_percentage:.3f}_{model_name}.csv"
    similarity_path = f"./data/{venue}/embedding/similarity.csv"

    # Initialize the review system
    review_system = PeerReviewSystem(
        comparison_model=args.comparison_model,
        generate=args.generate
    )

    # Construct
    if venue == "iclr_2025":
        dataset_path = f"./data/{venue}/iclr"
        papers = get_papers(dataset_path)
    else:
        other_data_path = f"./data/{venue}/submissions/submissions.csv"
        papers = get_other_papers(other_data_path)

    if not os.path.exists(paper_path):
        save_papers(papers, paper_path)

    # Generate
    if args.generate:
        for comparison_path, pair_type in zip(comparison_paths, pair_types):
            if not os.path.exists(comparison_path):
                # Generate Pairs
                review_system.generate_paper_pairs(
                    papers=papers,
                    comparison_path=comparison_path,
                    pair_type=pair_type,
                    similarity_path=similarity_path
                )
            # Call the LLM to generate comparative data
            review_system.generate_comparisons(
                comparison_path=comparison_path,
                batch_size=64
            )

    # Rank
    if args.rank:
        # Bradley-Terry Sorting
        review_system.bradley_terry_ranking(
            comparison_paths=comparison_paths,
            score_path=score_path,
            paper_path=paper_path,
            distribution_path=distribution_path,
            rank_method=args.rank_method,
            target_percentage=args.target_percentage,
            keep_number=args.keep_number,
            random_seed=args.keep_random_seed,
        )
        # Read scores and predicted results
        review_system.get_score(
            score_path=score_path,
        )


if __name__ == "__main__":
    main()

