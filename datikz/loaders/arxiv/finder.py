#!/usr/bin/env python

from collections import namedtuple
from functools import cache, cached_property
from os.path import join
from re import DOTALL, findall, finditer, search, sub
from subprocess import CalledProcessError, check_output

from TexSoup import TexSoup

from .demacro import Error as DemacroError, TexDemacro

# https://tex.stackexchange.com/a/115035
@cache
def listclasses():
    try:
        texmfdist = check_output(["kpsewhich", "--var-value=TEXMFDIST"])
        with open(join(texmfdist.strip(), b"ls-R")) as f:
            return [cls.removesuffix(".cls\n") for cls in f if cls.endswith(".cls\n")]
    except CalledProcessError:
        return list()

class TikzFinder():
    """
    Find tikzpictures and associated captions in a latex document and extract
    them as minimal compileable documents. Uses a combination of regex (fast)
    and TexSoup (slow) for searching.
    """
    Tikz = namedtuple("TikZ", ['code', 'caption'])
    Preamble = namedtuple("Preamble", ['imports', 'macros'])

    def __init__(self, tex):
        self.tex = self._check(tex.strip())

    def _check(self, tex):
        assert r"\documentclass" in tex, "No documentclass found!"
        assert r"\begin{document}" in tex, "No document found!"
        assert r"\end{document}" in tex, "File seems to be incomplete!"
        return tex

    @cached_property
    def _preamble(self) -> "Preamble":
        """
        Extract relevant package imports and possible macros from the document preamble.
        """
        # Patterns for the most common stuff to retain in a (tikz) document (\usepackage, \usetikzlibrary, \tikzset, etc).
        include = ["documentclass", "tikz", "tkz", "pgf"]
        # Patterns for other commonly used packages
        packages = ["inputenc", "fontenc", "fontspec", "amsmath", "amssymb", "color"]
        # hard exclude macros ([re]newcommand, [re]newenvironment), as they are handled by de-macro
        exclude = [r"\new", r"\renew"]
        preamble, *_ = self.tex.partition(r"\begin{document}")

        try:
            # try TexSoup first, as it works with multiline statements
            soup = TexSoup(preamble, tolerance=1)
            statements = map(str, soup.children)
        except:
            statements = preamble.split("\n")

        tikz_preamble, maybe_macros = list(), list()
        for stmt in statements:
            if not stmt.lstrip().startswith("%"): # filter line comments
                if (
                    not any(stmt.lstrip().startswith(pat) for pat in exclude)
                    and (
                        any(pat in stmt for pat in include)
                        or stmt.lstrip().startswith(r"\usepackage")
                        and any(pat in stmt for pat in packages)
                    )
                ):
                    if (
                        (cls:=listclasses())
                        and stmt.lstrip().startswith(r"\documentclass")
                        and not any(c in stmt for c in cls)
                    ):
                        # if the documentclass is not bundled with texlive use a fallback
                        tikz_preamble.append(r"\documentclass{article}")
                    else:
                        tikz_preamble.append(stmt)
                else:
                    maybe_macros.append(stmt)

        return self.Preamble(imports="\n".join(tikz_preamble).strip(), macros="\n".join(maybe_macros).strip())

    def _process_macros(self, macros, tikz, expand=True):
        try:
            ts = TexDemacro(macros=macros)
            return ts.process(tikz) if expand else "\n\n".join(ts.find(tikz)).strip()
        except (DemacroError, RecursionError, TypeError):
            return tikz if expand else ""

    def _find_colordefs(self, macros, tikz):
        definecolor_regex = r'^\s*\\definecolor(?:\[\w+?\])?\{(\w+?)\}\{\w+?\}\{.+?\}'

        matches = list()
        for color in finditer(definecolor_regex, macros):
            name, definition = color.group(1), color.group().lstrip()
            if search(rf"\b{name}\b", tikz):
                matches.append(definition)

        return "\n".join(matches).strip()

    def _replace_images(self, tikz):
        graphicx_regex = r'(\\includegraphics\*?(?:\[.*?\]){0,2})\{.+?\}'
        return sub(graphicx_regex, r'\1{example-image}', tikz)

    def _make_document(self, tikz: str) -> str:
        # substitute external image files with dummy images
        tikz = self._replace_images(tikz)
        # if the tikzpicture uses some macros, append them to the tikz preamble
        macros = self._process_macros(self._preamble.macros, tikz, expand=False)
        # also search for utilized color definitions
        colors = self._find_colordefs(self._preamble.macros, tikz)
        extended_preamble = self._preamble.imports + (f"\n\n{colors}" if colors else "") + (f"\n\n{macros}" if macros else "")

        return "\n\n".join([extended_preamble, r"\begin{document}", tikz, r"\end{document}"])

    def _clean_caption(self, caption: str) -> str:
        # expand any macros
        caption = self._process_macros(self._preamble.macros, caption)

        try:
            cap_soup = TexSoup(caption, tolerance=1)
            # remove any labels
            for label in cap_soup.find_all("label"):
                label.delete() # type: ignore
            caption = str(cap_soup)
        except:
            pass

        return " ".join(caption.split())

    def _find_caption(self, figure: str) -> str:
        """
        Captions need special handling, since we can't express balanced
        parentheses in regex.
        """
        (*_, raw_caption), caption, unmatched_parens = figure.partition(r"\caption{"), "", 1

        for c in raw_caption:
            if c == '}':
                unmatched_parens -= 1
            elif c == '{':
                unmatched_parens += 1
            if not unmatched_parens:
                break
            caption += c

        return caption

    def find(self):
        found = set()
        figure_regex = r"\\begin{figure}(.*?)\\end{figure}"
        tikz_regex = r"\\begin{tikzpicture}.*?\\end{tikzpicture}"
        # try to extract tikzpictures with captions first
        for figure in findall(figure_regex, self.tex, DOTALL):
            # extracting captions for figures with multiple tikzpictures (e.g. subfig) are above my paygrade
            if figure.count(r"\begin{tikzpicture}") == 1:
                if tikz := search(tikz_regex, figure, DOTALL):
                    if caption := self._find_caption(figure):
                        found.add(tikz.group())
                        yield self.Tikz(self._make_document(tikz.group()), self._clean_caption(caption))
        for tikz in findall(tikz_regex, self.tex, DOTALL):
            if tikz not in found:
                found.add(tikz)
                yield self.Tikz(self._make_document(tikz), '')

    def __call__(self, *args, **kwargs):
        yield from self.find(*args, **kwargs)
