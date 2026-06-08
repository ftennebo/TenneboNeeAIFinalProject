import matplotlib.pyplot as plt


def main():
    factors = [
        "Genre Overlap",
        "Content Similarity",
        "Collaborative Similarity",
        "Average Rating",
        "Popularity",
    ]

    weights = [0.40, 0.25, 0.20, 0.10, 0.05]
    percentages = [w * 100 for w in weights]

    colors = [
        "#4C78A8",
        "#59A14F",
        "#F28E2B",
        "#E15759",
        "#B07AA1",
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(factors, percentages, color=colors)

    plt.title("Hybrid Recommendation Score Weights", fontsize=16)
    plt.ylabel("Weight in Final Score (%)", fontsize=12)
    plt.ylim(0, 50)
    plt.xticks(rotation=15, ha="right")

    for bar, pct in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig("weighted_score_bar_chart.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
