from src.utils.seed import seed_everything
from src.utils.paths import PROJECT_ROOT

if __name__ == "__main__":
    seed_everything(42)
    print("Seed OK, root =", PROJECT_ROOT)
