"""Test làm sạch text trước TTS: bỏ markdown + emoji (TTS không đọc "sao sao")."""

import queue

from s2s_vn.LLM.lm_output_processor import LMOutputProcessor
from s2s_vn.pipeline.messages import EndOfResponse, LLMResponseChunk, TTSInput


def make_proc():
    iq, oq, tq = queue.Queue(), queue.Queue(), queue.Queue()
    p = LMOutputProcessor(iq, oq, text_out=tq)
    return p, oq


def feed_and_tts(p, oq, text):
    p.process(LLMResponseChunk(text_delta=text, turn_id=1))
    out = p.process(EndOfResponse(turn_id=1))
    return out


def test_markdown_stripped():
    p, oq = make_proc()
    out = feed_and_tts(p, oq, "Gói Pro có giá **99 USD** mỗi tháng.")
    assert isinstance(out, TTSInput)
    assert "**" not in out.text
    assert out.text == "Gói Pro có giá 99 USD mỗi tháng."


def test_markdown_variants_stripped():
    p, oq = make_proc()
    out = feed_and_tts(p, oq, "Nhiều *cách* #ghi `chú` ~xóa~ và _gạch_")
    assert "**" not in out.text and "*" not in out.text and "#" not in out.text
    assert "`" not in out.text and "~" not in out.text and "_" not in out.text


def test_emoji_stripped():
    p, oq = make_proc()
    out = feed_and_tts(p, oq, "Cảm ơn bạn! 😊👍")
    assert "😊" not in out.text
    assert "👍" not in out.text
    assert "Cảm ơn bạn!" in out.text


def test_vietnamese_and_punctuation_kept():
    p, oq = make_proc()
    out = feed_and_tts(p, oq, "Xin chào, tôi khỏe! Giá là 99 USD/tháng?")
    assert out.text == "Xin chào, tôi khỏe! Giá là 99 USD/tháng?"


def test_clean_does_not_touch_empty():
    p, oq = make_proc()
    out = feed_and_tts(p, oq, "")
    assert out is None
