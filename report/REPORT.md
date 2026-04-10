# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Nó có nghĩa là hai đoạn văn bản có nghĩa hoặc chủ đề cực kỳ giống nhau, làm cho góc giữa hai vector sinh ra từ chúng tiệm cận về 0 (chuẩn hóa cosine tiến về 1).

**Ví dụ HIGH similarity:**
- Sentence A: "Apple is a tech company."
- Sentence B: "Apple creates technology products."
- Tại sao tương đồng: Cả hai câu đều nói về một công ty công nghệ tạo ra sản phẩm. Context và các keyword gần như giống nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Apple is a tech company."
- Sentence B: "I like to eat red apples."
- Tại sao khác: Một câu nói về một công ty công nghệ (Apple Inc.), câu kia nói về việc ăn một loại trái cây (quả táo). Khác hẳn nhau về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Euclidean distance phụ thuộc vào độ dài vector (kích thước văn bản), do đó hai đoạn trùng về ý nghĩa nhưng dài ngắn khác nhau sẽ bị coi là khác biệt. Cosine similarity chuẩn hóa điều này và chỉ xét góc sinh ra giữa 2 vector, tập trung hoàn toàn vào sự tương đồng nội dung.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap)) = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Nếu overlap tăng lên 100 (tử số không đổi, mẫu số từ 450 thành 400), chunk count sẽ trở thành 25. Ta cần có overlap nhiều hơn để ngăn ý nghĩa câu văn bị gãy gọn ở phần ghép nối, tăng cường tính Semantic Completeness.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Customer support FAQ & AI Technical Docs

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* [Tùy chọn mô tả của nhóm bạn...]

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 01_faq_hoc_vu.txt | data/ | 10478 | source: ..., extension: .txt |
| 2 | 02_quy_che_sinh_vien_ktx.txt | data/ | 9063 | source: ..., extension: .txt |
| 3 | 03_huong_dan_hoc_bong.txt | data/ | 8460 | source: ..., extension: .txt |
| 4 | 04_thuc_tap_khoa_luan_tot_nghiep.txt | data/ | 9870 | source: ..., extension: .txt |
| 5 | 05_thu_vien_va_dich_vu_ho_tro.txt | data/ | 10443 | source: ..., extension: .txt |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | data/python_intro.txt | Trích xuất được xuất xứ để báo cho RAG Agent. |
| extension | string | .md, .txt | Khả chiếu nội dung dành cho dev hay content text thông thường. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| SAMPLE_TEXT | FixedSizeChunker (`fixed_size`) | 3 | 95.33 | Không giữ tốt. Chặt ngang chữ |
| SAMPLE_TEXT | SentenceChunker (`by_sentences`) | 2 | 122.50 | Tốt. Chặt theo ý nghĩa. |
| SAMPLE_TEXT | RecursiveChunker (`recursive`) | 4 | 60.00 | Tương đối. Phân nhỏ đoạn văn khá tốt. |

### Strategy Của Tôi

**Loại:** `SentenceChunker`

**Mô tả cách hoạt động:**
> *Viết 3-4 câu:* Sử dụng Regex Lookbehind để phân rã câu thông qua các dấu câu (. ! ?). Tiếp theo, tập hợp số lượng câu nhất định vào một mảng (chunk) theo tham số `max_sentences_per_chunk`. Việc này giúp không bị chặt khúc chữ cái giống `FixedSizeChunker` và bảo toàn ý tưởng tốt.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu:* [Tùy chọn: Vì FAQ/Playbook gồm nhiều câu lệnh, các lời dặn nhân viên được ngắt bằng dấu câu...]

**Code snippet (nếu custom):**
```python
import re

def chunk(self, text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    for i in range(0, len(sentences), self.max_sentences_per_chunk):
        chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker | 9/10 | Giữ semantic cực tốt | Chunk length đôi khi quá lớn nếu câu quá dàia |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* [Tùy nhóm kết luận...]

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?* Dùng Lookbehind regex: `r'(?<=[.!?])\s+'` để tách câu nhưng giữ lại dấu câu thuộc về câu trước nó. Khắc phục được lỗi cắt mất dấu phẩy/chấm ở cuối và gộp chính xác `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?* Thuật toán sẽ đệ quy mảng `separators`. Base case: Khi string bị split có chiều dài <= chunk_size thì trả về, hoặc khi không còn separators nào (`remaining_separators == []`) thì chia fix size tại chỗ.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?* Hỗ trợ dual-mode. Nếu có thư viện ChromaDB thì dùng `Collection`, nếu không sẽ lưu records vào list in-memory. Hàm tính similarity là thuật toán Dot Product (Cosine giữa query embedding vector và document embedding vector). Rank top_k theo score giảm dần.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?* `search_with_filter` thiết kế filter trước, loại bỏ list documents không mapping metadata (`{"key": "value"}`) rồi dùng list đó đưa qua search. `delete_document` duyệt array xoá hết các id matching với `doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?* Gọi `search()` từ Store để lấy top-k records, map lấy key `content` và gộp lại chung với string "Context:". Gửi vào LLM cùng user question với một lời cảnh báo *"If you don't know the answer, just say that you don't know"*.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.10s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | `[1.0, 0.0, 0.0]` | `[0.9, 0.1, 0.0]` | high | 0.993 | Yes |
| 2 | `[1.0, 0.0, 0.0]` | `[0.0, 1.0, 0.0]` | low | 0.00 | Yes |
| 3 | `[1.0, 0.0]` | `[-1.0, 0.0]` | low | -1.0 | Yes |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Điểm tương đồng của 2 vector trực giao mang kết quả Cosine chuẩn bằng `0.0`. Điều này cho thấy Embeddings sử dụng không gian n-chiều để tạo khoảng cách cực kỳ xa (hướng vector chênh nhau 90 độ) khi hai từ biểu thị định nghĩa khác nhau hoặc khi chênh nhau một lượng nhỏ cũng sẽ phản ánh mức độ (0.993 cho xảo nhỉnh hệ số).

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Summarize the key information from the loaded files. | ... |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Summarize the key information | "Customer Support Playbook..." | 0.195 | Yes | [DEMO LLM] Generated answer from prompt preview... |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* [Tùy chọn...]

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* [Tùy chọn...]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ định hướng metadata scheme có thêm trọng số Date, vì AI Agent cực kì dễ nhầm lẫn tài liệu chính sách cũ và mới.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |

