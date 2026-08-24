package vn.s2s.robotclient.audio;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

/**
 * Thu tiếng từ mic robot và đẩy ra ngoài theo từng chunk.
 *
 * <p>16kHz mono PCM16 — bằng đúng sample rate của pipeline nên không phải resample
 * ({@code talk.py:23-24}). Chunk 1280 mẫu = 80ms, giống client tham chiếu
 * ({@code talk.py:25}).
 *
 * <p>Nguồn âm dùng {@code VOICE_RECOGNITION}: nó bật chuỗi xử lý tiếng nói của thiết bị
 * (khử nhiễu) mà không ép sample rate như {@code VOICE_COMMUNICATION}.
 *
 * <p>Có đo biên độ và ghi log: khi thiếu quyền {@code RECORD_AUDIO} hoặc mic bị tiến
 * trình khác chiếm, {@link AudioRecord} vẫn chạy bình thường nhưng trả toàn số 0 — không
 * hề báo lỗi. Nhìn log biên độ là phân biệt được ngay hai ca đó với ca thu tốt.
 */
public class MicRecorder {

    private static final String TAG = "MicRecorder";

    /** Sample rate của pipeline s2s-vn — không đổi. */
    public static final int SAMPLE_RATE = 16000;

    /** 1280 mẫu = 2560 byte = 80ms, khớp CHUNK_SAMPLES của talk.py. */
    public static final int CHUNK_SAMPLES = 1280;

    private static final int CHUNK_BYTES = CHUNK_SAMPLES * 2;

    /** Nơi nhận audio thu được. Gọi trên luồng thu, không phải luồng UI. */
    public interface Listener {
        void onChunk(byte[] pcm);

        /**
         * Biên độ đỉnh của chunk vừa thu (0..32767), báo cho mọi chunk.
         *
         * <p>Luôn bằng 0 nghĩa là mic câm — thiếu quyền hoặc bị tiến trình khác chiếm.
         * Cả hai ca đó {@link AudioRecord} đều không báo lỗi gì.
         */
        void onBienDo(int bienDoDinh);

        /** Không mở được mic — thường do thiếu quyền hoặc tiến trình khác đang giữ. */
        void onError(String thongDiep);
    }

    private final Listener listener;
    private volatile boolean dangThu;
    private Thread luongThu;

    public MicRecorder(Listener listener) {
        this.listener = listener;
    }

    public boolean dangThu() {
        return dangThu;
    }

    /** Mở mic và bắt đầu đẩy chunk. Gọi lại khi đang thu thì không có tác dụng gì. */
    public void batDau() {
        if (dangThu) {
            return;
        }
        dangThu = true;
        luongThu = new Thread(new Runnable() {
            @Override
            public void run() {
                vongThu();
            }
        }, "mic-thu");
        luongThu.start();
    }

    /** Đóng mic. Chờ luồng thu kết thúc để chắc chắn mic đã được nhả. */
    public void dungLai() {
        dangThu = false;
        if (luongThu != null) {
            try {
                luongThu.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            luongThu = null;
        }
    }

    private void vongThu() {
        int coToiThieu = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (coToiThieu == AudioRecord.ERROR || coToiThieu == AudioRecord.ERROR_BAD_VALUE) {
            listener.onError("thiết bị không hỗ trợ thu 16kHz mono PCM16");
            dangThu = false;
            return;
        }

        // Buffer rộng gấp 4 chunk để chịu được lúc hệ thống bận, tránh mất tiếng.
        int coBuffer = Math.max(coToiThieu, CHUNK_BYTES * 4);
        AudioRecord rec = null;

        try {
            rec = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, coBuffer);

            if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
                listener.onError("không khởi tạo được AudioRecord (thiếu quyền RECORD_AUDIO?)");
                return;
            }

            rec.startRecording();
            Log.i(TAG, "bắt đầu thu: 16kHz mono, buffer " + coBuffer + " byte");

            byte[] chunk = new byte[CHUNK_BYTES];
            int soChunk = 0;

            while (dangThu) {
                int doc = rec.read(chunk, 0, CHUNK_BYTES);
                if (doc < 0) {
                    listener.onError("AudioRecord.read lỗi mã " + doc);
                    break;
                }
                if (doc == 0) {
                    continue;
                }

                byte[] guiDi = (doc == CHUNK_BYTES) ? chunk.clone()
                        : java.util.Arrays.copyOf(chunk, doc);
                listener.onChunk(guiDi);

                // Biên độ báo cho mọi chunk để dải sóng trên màn hình chạy mượt.
                int dinh = bienDoDinh(guiDi);
                listener.onBienDo(dinh);

                // Vẫn log mỗi ~1 giây: khi xem lại logcat sau sự cố thì cần có số.
                if (++soChunk % 12 == 0) {
                    Log.i(TAG, "chunk #" + soChunk + " biên độ đỉnh=" + dinh);
                }
            }
        } catch (Exception e) {
            listener.onError("mở mic thất bại: " + e.getMessage());
        } finally {
            if (rec != null) {
                try {
                    if (rec.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                        rec.stop();
                    }
                } catch (Exception ignored) {
                    // Thiết bị có thể đã ở trạng thái lỗi — vẫn phải release.
                }
                rec.release();
            }
            dangThu = false;
            Log.i(TAG, "đã dừng thu, nhả mic");
        }
    }

    /** Biên độ lớn nhất trong chunk (0..32767) — dùng để biết mic có nghe được gì không. */
    static int bienDoDinh(byte[] pcm16le) {
        int dinh = 0;
        for (int i = 0; i + 1 < pcm16le.length; i += 2) {
            int mau = (short) ((pcm16le[i] & 0xFF) | (pcm16le[i + 1] << 8));
            int abs = Math.abs(mau);
            if (abs > dinh) {
                dinh = abs;
            }
        }
        return dinh;
    }
}
