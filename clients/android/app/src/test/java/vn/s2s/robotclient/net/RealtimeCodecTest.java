package vn.s2s.robotclient.net;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Test;

/**
 * Test cho {@link RealtimeCodec} — mã hoá/giải mã giao thức WS của server s2s-vn.
 *
 * <p>Hợp đồng lấy từ client tham chiếu {@code src/s2s_vn/talk.py:85-106}.
 */
public class RealtimeCodecTest {

    /**
     * Mã hoá chunk audio thành message gửi lên, đúng khuôn {@code talk.py:85-86}.
     *
     * <p>Giá trị base64 mong đợi lấy độc lập bằng {@code base64.b64encode(bytes([1,2,3]))}
     * trong Python, không tính lại theo cách code làm.
     */
    @Test
    public void maHoaChunkAudioGuiLen() {
        String json = RealtimeCodec.encodeAudioAppend(new byte[] {1, 2, 3});

        assertEquals("{\"type\":\"input_audio_buffer.append\",\"audio\":\"AQID\"}", json);
    }

    /**
     * Base64 đúng cho mọi độ dài (padding 0/1/2 dấu {@code =}) và cho byte âm.
     *
     * <p>Giá trị mong đợi sinh độc lập bằng {@code base64.b64encode()} của Python, không
     * tính lại theo cách code làm. Ca {@code AID/fw==} quan trọng: chứa byte âm
     * ({@code 0x80}, {@code 0xFF} — Java không có kiểu byte không dấu) và ký tự {@code /},
     * đúng loại dữ liệu PCM16 thật gửi lên.
     */
    @Test
    public void base64DungChoMoiDoDaiVaByteAm() {
        assertEquals("", RealtimeCodec.base64Encode(new byte[] {}));
        assertEquals("AA==", RealtimeCodec.base64Encode(new byte[] {0}));
        assertEquals("AAE=", RealtimeCodec.base64Encode(new byte[] {0, 1}));
        assertEquals("AAEC", RealtimeCodec.base64Encode(new byte[] {0, 1, 2}));
        assertEquals("AAECAw==", RealtimeCodec.base64Encode(new byte[] {0, 1, 2, 3}));
        assertEquals("AAECAwQ=", RealtimeCodec.base64Encode(new byte[] {0, 1, 2, 3, 4}));
        assertEquals("AAECAwQF", RealtimeCodec.base64Encode(new byte[] {0, 1, 2, 3, 4, 5}));

        // PCM16 thật: byte âm + ký tự '/' trong kết quả.
        assertEquals("AID/fw==",
                RealtimeCodec.base64Encode(new byte[] {(byte) 0x00, (byte) 0x80, (byte) 0xFF, (byte) 0x7F}));
    }

    /** Nhận diện event báo model đã nạp xong — client chỉ được cho nói sau event này. */
    @Test
    public void nhanDienModelReady() {
        ServerEvent ev = RealtimeCodec.decode("{\"type\":\"server.model_ready\"}");

        assertEquals("server.model_ready", ev.type);
    }

    /**
     * Event audio: giải base64 field {@code delta} ra đúng PCM16 ({@code talk.py:100-102}).
     *
     * <p>{@code AID/fw==} là chuỗi Python sinh từ 4 byte {@code 00 80 FF 7F} — có byte âm
     * và ký tự {@code /}, đúng dạng audio thật.
     */
    @Test
    public void giaiMaAudioDelta() {
        ServerEvent ev = RealtimeCodec.decode(
                "{\"type\":\"response.output_audio.delta\",\"delta\":\"AID/fw==\"}");

        assertEquals("response.output_audio.delta", ev.type);
        assertArrayEquals(
                new byte[] {(byte) 0x00, (byte) 0x80, (byte) 0xFF, (byte) 0x7F}, ev.audio);
    }

    /** Event transcript: lấy đúng đoạn text robot đang đọc, giữ nguyên dấu tiếng Việt. */
    @Test
    public void giaiMaTranscriptDelta() {
        ServerEvent ev = RealtimeCodec.decode(
                "{\"type\":\"response.output_audio_transcript.delta\",\"delta\":\"Xin chào\"}");

        assertEquals("response.output_audio_transcript.delta", ev.type);
        assertEquals("Xin chào", ev.text);
    }

    /**
     * Event lỗi: lấy ra thông điệp đọc được, không phải cả cục JSON.
     *
     * <p>Server đóng gói lỗi thành object chứ không phải chuỗi
     * ({@code realtime_service.py:294-299}):
     * {@code {"type":"invalid_request_error","code":null,"message":"...","param":null}}.
     * Nhét nguyên object vào log thì người đọc phải tự bới — client cần lấy field
     * {@code message}.
     */
    @Test
    public void layThongDiepTuEventLoi() {
        ServerEvent ev = RealtimeCodec.decode(
                "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\","
                        + "\"code\":null,\"message\":\"malformed JSON\",\"param\":null}}");

        assertEquals("error", ev.type);
        assertEquals("malformed JSON", ev.text);
    }

    /**
     * Dữ liệu bẩn từ mạng không được làm sập client — trả {@code null} để bỏ qua.
     *
     * <p>Nếu một trong các ca này ném exception, luồng WebSocket sẽ chết và robot câm
     * mà không báo gì.
     */
    @Test
    public void duLieuBanTraNullChuKhongNemException() {
        assertNull("JSON hỏng", RealtimeCodec.decode("{khong phai json"));
        assertNull("chuỗi rỗng", RealtimeCodec.decode(""));
        assertNull("null", RealtimeCodec.decode(null));
        assertNull("thiếu field type", RealtimeCodec.decode("{\"delta\":\"AQID\"}"));
        assertNull("JSON array chứ không phải object", RealtimeCodec.decode("[1,2,3]"));

        // Event lạ (server thêm event mới) -> nhận được, không sập; client tự bỏ qua.
        ServerEvent la = RealtimeCodec.decode("{\"type\":\"event.hoan.toan.moi\"}");
        assertEquals("event.hoan.toan.moi", la.type);

        // Event audio mà base64 hỏng -> không sập, audio để null.
        ServerEvent hong = RealtimeCodec.decode(
                "{\"type\":\"response.output_audio.delta\",\"delta\":\"!!!khong-phai-base64!!!\"}");
        assertNull("base64 hỏng thì audio phải null", hong == null ? null : hong.audio);
    }
}
