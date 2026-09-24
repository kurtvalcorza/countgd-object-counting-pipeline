from vit_mae_pipeline import load_pipeline


def main() -> None:
    pipe = load_pipeline()
    result = pipe.reconstruct("photo.jpg", seed=0)
    entry = result["results"][0]
    print(f"hidden patches: {entry['hidden_patches']} of 196")
    print(f"masked MSE: {entry['masked_mse']:.4f}")
    print(f"the model's own loss: {result['model_loss']:.4f}")
    entry["reconstruction"].save("reconstruction.png")


if __name__ == "__main__":
    main()
