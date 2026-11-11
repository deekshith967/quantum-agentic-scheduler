"""Re-create the synthetic demo whiteboard photo used by the running scenario."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qas import config  # noqa: E402
from qas.perception.board_render import render_board  # noqa: E402

if __name__ == "__main__":
    p = render_board([("To Do", "Payment API integration"), ("To Do", "UI testing"), ("To Do", "Release notes"), ("Done", "Login bug fix")],
                     config.DEMO_DIR / "whiteboard.png")
    print("wrote", p)
