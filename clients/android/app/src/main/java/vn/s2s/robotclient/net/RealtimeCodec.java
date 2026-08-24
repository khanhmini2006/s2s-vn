package vn.s2s.robotclient.net;

/**
 * Mã hoá/giải mã giao thức WebSocket của server s2s-vn.
 *
 * <p>Hàm thuần, không trạng thái. Hợp đồng lấy từ client tham chiếu
 * {@code src/s2s_vn/talk.py:85-106}: mọi thứ đi qua text frame JSON, audio nhúng
 * dạng base64 — không dùng binary frame.
 */
public final class RealtimeCodec {

    // --- Tên event server phát ra ---

    /** Model đã nạp xong — client chỉ được cho phép nói sau event này. */
    public static final String EVENT_MODEL_READY = "server.model_ready";
    /** VAD server nghe thấy người nói — phải xoá ngay buffer đang phát (barge-in). */
    public static final String EVENT_SPEECH_STARTED = "input_audio_buffer.speech_started";
    /** Người dùng nói xong, server bắt đầu xử lý. */
    public static final String EVENT_SPEECH_STOPPED = "input_audio_buffer.speech_stopped";
    /** Một mẩu audio trả lời (PCM16 16kHz, base64 trong field {@code delta}). */
    public static final String EVENT_AUDIO_DELTA = "response.output_audio.delta";
    /** Một mẩu text của câu robot đang đọc. */
    public static final String EVENT_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta";
    /** Xong một lượt trả lời. */
    public static final String EVENT_RESPONSE_DONE = "response.done";
    /** Server báo lỗi. */
    public static final String EVENT_ERROR = "error";

    // --- Bảng base64 ---
    // Thứ tự khai báo hai field này quan trọng: static field khởi tạo theo thứ tự xuất
    // hiện trong file. Đặt NGUOC_BASE64 lên trước thì taoBangNguoc() đọc phải null và cả
    // lớp chết bằng ExceptionInInitializerError (đã dính lỗi này một lần).

    private static final char[] BANG_BASE64 =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".toCharArray();

    private static final int[] NGUOC_BASE64 = taoBangNguoc();

    private RealtimeCodec() {}

    /**
     * Gói một chunk audio PCM16 16kHz mono thành message gửi lên server.
     *
     * <p>Khuôn message theo {@code talk.py:85-86}.
     */
    public static String encodeAudioAppend(byte[] pcm) {
        return "{\"type\":\"input_audio_buffer.append\",\"audio\":\"" + base64Encode(pcm) + "\"}";
    }

    /**
     * Giải mã một text frame nhận từ server.
     *
     * @return event đã giải mã, hoặc {@code null} nếu không đọc được — dữ liệu đến từ
     *     mạng nên không bao giờ được tin; hỏng thì bỏ qua chứ không làm sập client.
     */
    public static ServerEvent decode(String json) {
        try {
            org.json.JSONObject o = new org.json.JSONObject(json);
            String type = o.optString("type", null);
            if (type == null || type.isEmpty()) {
                return null;
            }

            byte[] audio = null;
            String text = null;

            if (EVENT_AUDIO_DELTA.equals(type)) {
                audio = base64Decode(o.optString("delta", ""));
            } else if (EVENT_TRANSCRIPT_DELTA.equals(type)) {
                text = o.optString("delta", null);
            } else if (EVENT_ERROR.equals(type)) {
                // Server gói lỗi thành object, không phải chuỗi
                // (realtime_service.py:294-299) — lấy field "message" cho người đọc,
                // rơi về cả cục nếu khuôn khác dự kiến.
                Object loi = o.opt("error");
                if (loi instanceof org.json.JSONObject) {
                    text = ((org.json.JSONObject) loi).optString("message", loi.toString());
                } else if (loi != null) {
                    text = String.valueOf(loi);
                }
            }

            return new ServerEvent(type, audio, text);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Mã hoá base64 tự viết.
     *
     * <p>Không dùng {@code java.util.Base64} (cần API 26) hay {@code android.util.Base64}
     * (trong android.jar chỉ là stub, gọi trong JVM unit test sẽ ném {@code "Stub!"}) —
     * robot chạy minSdk 19 và seam này phải test được không cần thiết bị.
     */
    static String base64Encode(byte[] data) {
        StringBuilder sb = new StringBuilder((data.length + 2) / 3 * 4);
        int i = 0;

        while (i + 2 < data.length) {
            int ba = ((data[i] & 0xFF) << 16) | ((data[i + 1] & 0xFF) << 8) | (data[i + 2] & 0xFF);
            sb.append(BANG_BASE64[(ba >>> 18) & 0x3F]);
            sb.append(BANG_BASE64[(ba >>> 12) & 0x3F]);
            sb.append(BANG_BASE64[(ba >>> 6) & 0x3F]);
            sb.append(BANG_BASE64[ba & 0x3F]);
            i += 3;
        }

        int con = data.length - i;
        if (con == 1) {
            int ba = (data[i] & 0xFF) << 16;
            sb.append(BANG_BASE64[(ba >>> 18) & 0x3F]);
            sb.append(BANG_BASE64[(ba >>> 12) & 0x3F]);
            sb.append("==");
        } else if (con == 2) {
            int ba = ((data[i] & 0xFF) << 16) | ((data[i + 1] & 0xFF) << 8);
            sb.append(BANG_BASE64[(ba >>> 18) & 0x3F]);
            sb.append(BANG_BASE64[(ba >>> 12) & 0x3F]);
            sb.append(BANG_BASE64[(ba >>> 6) & 0x3F]);
            sb.append('=');
        }

        return sb.toString();
    }

    /**
     * Giải mã base64.
     *
     * @return mảng byte, hoặc {@code null} nếu chuỗi không hợp lệ — dữ liệu từ mạng
     *     không được tin.
     */
    static byte[] base64Decode(String s) {
        if (s == null) {
            return null;
        }

        int het = s.length();
        while (het > 0 && s.charAt(het - 1) == '=') {
            het--;
        }

        java.io.ByteArrayOutputStream ra = new java.io.ByteArrayOutputStream();
        int gom = 0;
        int soBit = 0;

        for (int i = 0; i < het; i++) {
            char c = s.charAt(i);
            if (c >= 128 || NGUOC_BASE64[c] < 0) {
                return null; // ký tự không thuộc bảng base64
            }
            gom = (gom << 6) | NGUOC_BASE64[c];
            soBit += 6;
            if (soBit >= 8) {
                soBit -= 8;
                ra.write((gom >>> soBit) & 0xFF);
            }
        }

        return ra.toByteArray();
    }

    private static int[] taoBangNguoc() {
        int[] bang = new int[128];
        java.util.Arrays.fill(bang, -1);
        for (int i = 0; i < BANG_BASE64.length; i++) {
            bang[BANG_BASE64[i]] = i;
        }
        return bang;
    }
}
