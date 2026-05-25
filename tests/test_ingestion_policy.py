from pathlib import Path
import unittest

from genai_lab.ingestion.chunking import ChunkingProfile
from genai_lab.ingestion.metadata import build_source_metadata, classify_source_type


class IngestionPolicyTests(unittest.TestCase):
    def test_chunking_profile_rejects_overlap_larger_than_chunk(self) -> None:
        profile = ChunkingProfile(chunk_size_tokens=256, chunk_overlap_tokens=256)

        with self.assertRaisesRegex(ValueError, "smaller"):
            profile.validate()

    def test_chunking_profile_reports_overlap_ratio(self) -> None:
        profile = ChunkingProfile(chunk_size_tokens=512, chunk_overlap_tokens=64)

        self.assertEqual(profile.overlap_ratio, 0.125)

    def test_source_metadata_is_filterable(self) -> None:
        metadata = build_source_metadata(Path("papers/example.md"))

        self.assertEqual(metadata["source_name"], "example.md")
        self.assertEqual(metadata["source_type"], "markdown")
        self.assertEqual(metadata["corpus"], "research")
        self.assertIn("ingested_at", metadata)

    def test_source_type_classification(self) -> None:
        self.assertEqual(classify_source_type(Path("paper.pdf")), "paper_pdf")
        self.assertEqual(classify_source_type(Path("notes.txt")), "text")
        self.assertEqual(classify_source_type(Path("unknown")), "unknown")


if __name__ == "__main__":
    unittest.main()
