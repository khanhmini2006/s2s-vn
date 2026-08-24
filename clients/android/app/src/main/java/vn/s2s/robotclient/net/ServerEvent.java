package vn.s2s.robotclient.net;

/**
 * Một event nhận từ server s2s-vn, đã giải mã.
 *
 * <p>Kiểu dữ liệu thuần, không hành vi. Tuỳ {@link #type} mà {@link #audio} hoặc
 * {@link #text} có giá trị.
 */
public class ServerEvent {

    /** Tên event, ví dụ {@code "server.model_ready"}. */
    public final String type;

    /** PCM16 16kHz đã giải base64 — chỉ có với {@code response.output_audio.delta}. */
    public final byte[] audio;

    /** Đoạn text — chỉ có với các event mang transcript hoặc lỗi. */
    public final String text;

    public ServerEvent(String type, byte[] audio, String text) {
        this.type = type;
        this.audio = audio;
        this.text = text;
    }
}
