package vn.s2s.robotclient.net;

import java.net.URI;

import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;

/**
 * Kết nối WebSocket tới server s2s-vn, tự nối lại khi đứt.
 *
 * <p>Giao thức xem {@link RealtimeCodec}. Mọi callback trong {@link Listener} chạy trên
 * luồng của thư viện WebSocket, KHÔNG phải luồng UI — Activity phải tự chuyển về main
 * thread trước khi đụng view.
 *
 * <p>Nối lại dùng backoff tăng dần vì mỗi phiên WS làm server dựng một pipeline kèm
 * một bản model riêng, chỉ dọn khi đứt kết nối hoặc sau 300s không có event
 * ({@code websocket_router.py:60}). Nối lại dồn dập sẽ làm cạn VRAM của server.
 */
public class RealtimeWsClient {

    /** Nơi nhận sự kiện từ server. Mọi phương thức chạy ngoài luồng UI. */
    public interface Listener {
        /** Đã nối được (chưa chắc model đã sẵn sàng — chờ {@link #onModelReady()}). */
        void onConnected();

        /** Model đã nạp xong; từ lúc này mới nên cho người dùng nói. */
        void onModelReady();

        /** VAD server nghe thấy người nói — phải xoá ngay audio đang phát. */
        void onSpeechStarted();

        /** Người dùng nói xong, server bắt đầu xử lý — đây là lúc đóng mic. */
        void onSpeechStopped();

        /** Một mẩu audio trả lời (PCM16 16kHz) — đẩy vào bộ đệm phát. */
        void onAudioChunk(byte[] pcm);

        /** Một mẩu text của câu robot đang đọc. */
        void onTranscript(String text);

        /** Xong một lượt trả lời. */
        void onResponseDone();

        /** Mất kết nối; {@code sePhutNoiLai} là số giây trước lần thử lại kế tiếp. */
        void onDisconnected(String lyDo, int sePhutNoiLai);

        /** Lỗi từ server hoặc từ tầng mạng. */
        void onError(String thongDiep);
    }

    private static final int BACKOFF_DAU_S = 1;
    private static final int BACKOFF_TOI_DA_S = 30;

    private final URI uri;
    private final Listener listener;

    private WebSocketClient ws;
    private Thread luongNoiLai;
    private volatile boolean dangMuonNoi;
    private volatile boolean modelSanSang;
    private int backoffS = BACKOFF_DAU_S;

    public RealtimeWsClient(String url, Listener listener) {
        this.uri = URI.create(url);
        this.listener = listener;
    }

    /** Đã nối và server báo model nạp xong. */
    public boolean sanSangNoi() {
        return modelSanSang && ws != null && ws.isOpen();
    }

    /** Bắt đầu nối, và tự nối lại mỗi khi đứt cho tới khi gọi {@link #dong()}. */
    public void noi() {
        dangMuonNoi = true;
        moKetNoi();
    }

    /** Dừng hẳn: không nối lại nữa và đóng kết nối hiện tại. */
    public void dong() {
        dangMuonNoi = false;
        modelSanSang = false;
        if (luongNoiLai != null) {
            luongNoiLai.interrupt();
            luongNoiLai = null;
        }
        if (ws != null) {
            ws.close();
            ws = null;
        }
    }

    /**
     * Gửi một chunk audio lên server.
     *
     * <p>Bỏ qua lặng lẽ nếu chưa nối được — mic có thể vẫn đang chạy lúc mạng vừa đứt,
     * và ném exception ở đây sẽ giết luồng thu.
     */
    public void guiAudio(byte[] pcm) {
        WebSocketClient hienTai = ws;
        if (hienTai != null && hienTai.isOpen()) {
            try {
                hienTai.send(RealtimeCodec.encodeAudioAppend(pcm));
            } catch (Exception e) {
                // Kết nối vừa đứt giữa lúc kiểm tra và lúc gửi — coi như mất chunk này.
            }
        }
    }

    private void moKetNoi() {
        modelSanSang = false;
        ws = new WebSocketClient(uri) {
            @Override
            public void onOpen(ServerHandshake handshake) {
                backoffS = BACKOFF_DAU_S; // nối được rồi thì reset thang backoff
                listener.onConnected();
            }

            @Override
            public void onMessage(String message) {
                xuLyTinNhan(message);
            }

            @Override
            public void onClose(int code, String reason, boolean remote) {
                modelSanSang = false;
                henNoiLai(reason == null || reason.isEmpty() ? "đóng (mã " + code + ")" : reason);
            }

            @Override
            public void onError(Exception ex) {
                listener.onError(String.valueOf(ex.getMessage()));
                // Không tự nối lại ở đây: sau onError thư viện luôn gọi onClose.
            }
        };
        ws.setConnectionLostTimeout(60);
        ws.connect();
    }

    private void xuLyTinNhan(String message) {
        ServerEvent ev = RealtimeCodec.decode(message);
        if (ev == null) {
            return; // dữ liệu bẩn từ mạng — bỏ qua, không làm chết luồng
        }

        if (RealtimeCodec.EVENT_MODEL_READY.equals(ev.type)) {
            modelSanSang = true;
            listener.onModelReady();
        } else if (RealtimeCodec.EVENT_SPEECH_STARTED.equals(ev.type)) {
            listener.onSpeechStarted();
        } else if (RealtimeCodec.EVENT_SPEECH_STOPPED.equals(ev.type)) {
            listener.onSpeechStopped();
        } else if (RealtimeCodec.EVENT_AUDIO_DELTA.equals(ev.type)) {
            if (ev.audio != null) {
                listener.onAudioChunk(ev.audio);
            }
        } else if (RealtimeCodec.EVENT_TRANSCRIPT_DELTA.equals(ev.type)) {
            if (ev.text != null) {
                listener.onTranscript(ev.text);
            }
        } else if (RealtimeCodec.EVENT_RESPONSE_DONE.equals(ev.type)) {
            listener.onResponseDone();
        } else if (RealtimeCodec.EVENT_ERROR.equals(ev.type)) {
            listener.onError(ev.text == null ? "lỗi không rõ" : ev.text);
        }
        // Event khác (session.created, ...) bỏ qua — client này không cần.
    }

    private void henNoiLai(final String lyDo) {
        if (!dangMuonNoi) {
            listener.onDisconnected(lyDo, 0);
            return;
        }

        final int cho = backoffS;
        listener.onDisconnected(lyDo, cho);
        backoffS = Math.min(backoffS * 2, BACKOFF_TOI_DA_S);

        luongNoiLai = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(cho * 1000L);
                } catch (InterruptedException e) {
                    return; // dong() đã được gọi
                }
                if (dangMuonNoi) {
                    moKetNoi();
                }
            }
        }, "ws-noi-lai");
        luongNoiLai.start();
    }
}
