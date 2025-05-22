import os
import nbformat
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "./data"
OUTPUT_DIR = os.path.join(DATA_DIR, "auto_summaries")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def summarize_notebook(path):
    try:
        nb = nbformat.read(path, as_version=4)
        texts = []
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                texts.append(cell.source)
            elif cell.cell_type == "code":
                # Extract comments in code cells for insights
                lines = cell.source.splitlines()
                comments = [line for line in lines if line.strip().startswith("#")]
                texts.extend(comments)
        summary = "\n\n".join(texts)
        return summary
    except Exception as e:
        return f"Failed to summarize notebook: {e}"

def summarize_csv(path):
    try:
        df = pd.read_csv(path, nrows=100)
        summary = f"File: {os.path.basename(path)}\n"
        summary += f"Columns: {', '.join(df.columns)}\n\n"
        summary += f"Sample data:\n{df.head(5).to_string(index=False)}\n\n"
        summary += f"Basic stats:\n{df.describe(include='all').to_string()}"
        return summary
    except Exception as e:
        return f"Failed to summarize CSV: {e}"

def summarize_html(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            texts = soup.get_text(separator="\n").strip()
            # Take first 1000 chars approx as summary
            summary = texts[:1000] + "\n...[truncated]"
        return summary
    except Exception as e:
        return f"Failed to summarize HTML: {e}"

def main():
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            filename, ext = os.path.splitext(file)
            ext = ext.lower()
            if ext == ".ipynb":
                print(f"Summarizing notebook: {file}")
                summary = summarize_notebook(file_path)
            elif ext == ".csv":
                print(f"Summarizing CSV: {file}")
                summary = summarize_csv(file_path)
            elif ext == ".html":
                print(f"Summarizing HTML: {file}")
                summary = summarize_html(file_path)
            else:
                continue

            output_path = os.path.join(OUTPUT_DIR, f"{filename}_summary.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)

    print(f"\n✅ Summaries saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
