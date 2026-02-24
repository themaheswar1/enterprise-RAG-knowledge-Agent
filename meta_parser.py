"""
ReadMe :
This file is a custom PDF Parser for Agent Extracts text line by line and chunks
with full positional metadata like: doc_id, page_num, line_start, line_end,
chunk_index and finally text.

"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import pdfplumber

@dataclass
class Line:
    #For a single line extracted from pdf
    doc_id: str
    page_num: int
    line_num: int
    page_line_num: int
    text: str

@dataclass
class Chunk:
    #For a text chunk ready for embedding, with full metadata
    chunk_id: str   #unique hash
    doc_id: str
    filename: str
    page_num: int
    line_start: int
    line_end: int
    page_line_start: int
    page_line_end: int
    chunk_index: int
    text: str

    def to_metadata(self) -> dict:
        # Returns everything except text - used as ChromaDB metadata
        return {
            "chunk_id":       self.chunk_id,
            "doc_id":         self.doc_id,
            "filename":       self.filename,
            "page_num":       self.page_num,
            "line_start":     self.line_start,
            "line_end":       self.line_end,
            "page_line_start": self.page_line_start,
            "page_line_end":  self.page_line_end,
            "chunk_index":    self.chunk_index,
        }
    
    def citation(self)  -> str:
        # Human-readable citation string
        return {
            f' file is: {self.filename}'
            f' Page is: {self.page_num}'
            f' Lines are: {self.line_start}-{self.page_line_end}'
            
        }
# Core Parser for our data
class PDFParser:
    def __init__(
            self,
            chunk_size: int = 20,
            overlap: int = 4,
            min_line_length: int = 3,
            ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_line_length = min_line_length

    def extract_lines(self, pdf_path: str) -> list[Line]:
           path = Path(pdf_path)
           doc_id = self._make_doc_id(path)
           lines = []
           global_line_num = 1

           with pdfplumber.open(pdf_path) as pdf:
               for page_index, page in enumerate(pdf.pages):
                   page_num = page_index + 1
                   raw_text = page.extract_text(x_tolerance = 2, y_tolerance=3)

                   if not raw_text:
                       continue
                   
                   raw_lines = raw_text.split("\n")
                   page_line_num = 1

                   for raw_line in raw_lines:
                       cleaned = self.clean_line(raw_line)
                       if len(cleaned) < self.min_line_length:
                           continue
                       
                       lines.append(Line(
                           doc_id=doc_id,
                           page_num=page_num,
                           line_num=global_line_num,
                           page_line_num=page_line_num,
                           text=cleaned
                        
                       ))
                       global_line_num += 1
                       page_line_num += 1

           return lines  

    def chunk_lines(self, lines: list[Line], filename: str) -> list[Chunk]:

        # lines to chunks
        if not lines:
            return []
        
        doc_id = lines[0].doc_id
        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0
        i = 0

        while i < len(lines):
            window = lines[i: i + self.chunk_size]
            if not window:
                break

            text = "\n".join(line.text for line in window)
            first = window[0]
            last = window[-1]

            chunk_id = self._make_chunk_id(doc_id,first.line_num,last.line_num)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                filename=filename,
                page_num=first.page_num,
                line_start=first.line_num,
                line_end=last.line_num,
                page_line_start=first.page_line_num,
                page_line_end=last.page_line_num,
                chunk_index=chunk_index,
                text=text,
            ))

            chunk_index += 1
            i += step

        return chunks

    # --Main entry point--
    def parse(self, pdf_path: str) -> list[Chunk]:
        # pdf file -> list of chunks with metadata
        # This is what ingest.py will call
        path = Path(pdf_path)
        print(f" Parsing: {path.name} ")

        lines = self.extract_lines(pdf_path)
        print(f"    -> {len(lines)} lines extracted")

        chunks = self.chunk_lines(lines, filename=path.name)
        print(f"   -> {len(chunks)} chunks created"
              f" (size = {self.chunk_size}, overlap= {self.overlap})")

        return chunks

    # -- Helpers --
    def clean_line(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)       
        text = re.sub(r'[^\x20-\x7E\n]', '', text)  
        return text

    def _make_doc_id(self, path: Path) -> str:
        return hashlib.md5(path.name.encode()).hexdigest()[:12]

    def _make_chunk_id(self, doc_id: str, line_start: int, line_end: int) -> str:
        raw = f"{doc_id}:{line_start}:{line_end}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
    
# ─────────────────────────────────────────────
# Standalone test — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        # Default: test against first PDF in ./docs/
        docs = list(Path("docs").glob("*.pdf"))
        if not docs:
            print("No PDFs found in ./docs/ — pass a path as argument.")
            sys.exit(1)
        pdf_path = str(docs[0])
    else:
        pdf_path = sys.argv[1]

    parser = PDFParser(chunk_size=20, overlap=4)
    chunks = parser.parse(pdf_path)

    print(f"\n{'─'*60}")
    print(f"SAMPLE CHUNKS from: {Path(pdf_path).name}")
    print(f"{'─'*60}")

    # Show first 3 chunks
    for chunk in chunks[:3]:
        print(f"\n[Chunk {chunk.chunk_index}]")
        print(f"  Citation : {chunk.citation()}")
        print(f"  Lines    : {chunk.line_start}–{chunk.line_end} (global)")
        print(f"  Page     : {chunk.page_num}")
        print(f"  Text     :\n    " + chunk.text[:300].replace("\n", "\n    "))

    print(f"\n{'─'*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Metadata keys: {list(chunks[0].to_metadata().keys())}")    
                   


        






