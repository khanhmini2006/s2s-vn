import os
import shutil
import unittest

from src.s2s_vn.api.rag_service import RAGService


class TestRAGService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a temporary directory for testing ChromaDB
        cls.test_db_dir = "./test_chroma_db"
        if os.path.exists(cls.test_db_dir):
            shutil.rmtree(cls.test_db_dir)
        cls.rag_service = RAGService(persist_directory=cls.test_db_dir)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary database
        if os.path.exists(cls.test_db_dir):
            shutil.rmtree(cls.test_db_dir)

    def test_add_and_search_document(self):
        doc_text = (
            "Dự án s2s-vn là một trợ lý giọng nói mã nguồn mở dành cho tiếng Việt. "
            "Nó sử dụng PhoWhisper cho STT, Qwen cho LLM và VieNeu cho TTS. "
            "Kiến trúc dựa trên luồng xử lý thời gian thực với độ trễ thấp."
        )
        
        # Thêm document
        num_chunks = self.rag_service.add_document(doc_text, doc_id="test_doc_1")
        self.assertGreater(num_chunks, 0)
        
        # Tìm kiếm query có liên quan
        results = self.rag_service.search("s2s-vn dùng model nào cho TTS?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("VieNeu", results[0])
        
        # Tìm kiếm truy vấn không liên quan
        # Mặc dù query không liên quan, ChromaDB vẫn trả về chunk gần nhất theo vector similarity, 
        # nhưng ta chỉ test nó vẫn chạy không lỗi.
        results_irrelevant = self.rag_service.search("thời tiết hôm nay thế nào?", top_k=1)
        self.assertEqual(len(results_irrelevant), 1)

    def test_chunking(self):
        text = "A" * 600
        chunks = self.rag_service._chunk_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 500)
        self.assertEqual(len(chunks[1]), 150) # 100 char mới + 50 overlap

if __name__ == "__main__":
    unittest.main()
