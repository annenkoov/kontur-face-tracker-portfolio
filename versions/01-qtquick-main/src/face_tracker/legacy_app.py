"""Fallback launcher for the previous Tk interface during Qt migration."""

import tkinter as tk

from .ui import TrackerApp


def main() -> None:
    root = tk.Tk()
    TrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
