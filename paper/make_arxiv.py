"""Extract paper/ into a self-contained folder ready to upload to arXiv.

    python paper/make_arxiv.py            # -> arxiv_submission/ and arxiv_submission.tar.gz

The folder holds main.tex, neurips_2026.sty, main.bbl and the figures the
paper actually includes, flattened into figs/.  arXiv publishes the source,
so every LaTeX comment is stripped from main.tex.  arXiv does not run BibTeX,
so the .bbl from a fresh build of paper/ ships in its place.  The folder is
then compiled from scratch to check that it builds on its own.
"""

import argparse
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent  # repository root; \graphicspath{{../}} resolves figures here


def strip_comments(tex: str) -> str:
    """Remove LaTeX comments without changing how the document typesets.

    A line that is only a comment is dropped outright (leaving a blank line
    would start a new paragraph).  A trailing comment is cut back to a bare
    `%`, which keeps its job of swallowing the end-of-line space (as in
    `\\author{%`).  A `%` counts as a comment only when it is preceded by an
    even number of backslashes, so `\\%` survives and `\\\\%` does not.
    """
    out = []
    for line in tex.splitlines():
        cut = None
        for m in re.finditer("%", line):
            backslashes = len(line[: m.start()]) - len(line[: m.start()].rstrip("\\"))
            if backslashes % 2 == 0:
                cut = m.start()
                break
        if cut is None:
            out.append(line)
        elif line[:cut].strip() == "":
            continue
        else:
            out.append(line[:cut].rstrip() + "%")
    # Collapse the runs of blank lines left where commented blocks used to be.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def included_graphics(tex: str) -> list[str]:
    """Paths passed to \\includegraphics, in order, after comments are gone."""
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)


def latexmk(folder: Path) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=folder, check=True, stdout=subprocess.DEVNULL,
    )


def pdf_pages(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
    return int(re.search(r"^Pages:\s+(\d+)", info.stdout, re.M).group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "arxiv_submission")
    args = ap.parse_args()
    out: Path = args.out

    # Rebuild paper/ first so the .bbl matches the current main.tex.
    latexmk(PAPER)

    tex = strip_comments((PAPER / "main.tex").read_text())
    # Figures move next to main.tex, so the ../ search path goes, and
    # \pdfoutput=1 up front tells arXiv to run pdfLaTeX (the figures are PDFs).
    tex = re.sub(r"\\graphicspath\{\{\.\./\}\}%?\n", "", tex)
    tex = "\\pdfoutput=1\n" + tex

    if out.exists():
        shutil.rmtree(out)
    (out / "figs").mkdir(parents=True)
    (out / "main.tex").write_text(tex)
    shutil.copy2(PAPER / "neurips_2026.sty", out)
    shutil.copy2(PAPER / "main.bbl", out)
    for fig in included_graphics(tex):
        shutil.copy2(ROOT / fig, out / fig)

    # Check the folder builds on its own, then drop the build products
    # (keeping main.bbl, which arXiv needs).  With no .bib in the folder,
    # latexmk skips BibTeX and uses the shipped main.bbl, as arXiv will.
    latexmk(out)
    pages = {d.name: pdf_pages(d / "main.pdf") for d in (PAPER, out)}
    if pages[PAPER.name] != pages[out.name]:
        raise SystemExit(f"page count changed in extraction: {pages}")
    for f in out.iterdir():
        if f.is_file() and f.name not in {"main.tex", "main.bbl", "neurips_2026.sty"}:
            f.unlink()

    tarball = out.with_suffix(".tar.gz")
    with tarfile.open(tarball, "w:gz") as tar:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=f.relative_to(out))

    n_files = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"{out}: {n_files} files, builds to {pages[out.name]} pages")
    print(f"upload: {tarball}")


if __name__ == "__main__":
    main()
